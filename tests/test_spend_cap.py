"""
Tests for LLM spend accounting and enforcement.

Covers the three holes that made "no spend cap" the real production risk:
  1. Cost was priced at Sonnet rates while the default model is Haiku.
  2. Four of seven messages.create() sites never recorded their cost at all.
  3. Guest identities and a client-supplied X-Forwarded-For could reset every
     per-caller limit at will.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agents import pricing, spend
from backend.api import budget, run_manager


def _usage(inp=0, out=0, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


@pytest.fixture(autouse=True)
def _reset_budget():
    budget.reset_local_budget()
    yield
    budget.reset_local_budget()


# ── Pricing ───────────────────────────────────────────────────────────────────


def test_haiku_priced_as_haiku_not_sonnet():
    """The tracer hardcoded $3/$15 while FAST_MODEL defaults to Haiku."""
    inp, out, _, _ = pricing.rates("claude-haiku-4-5-20251001")
    assert (inp, out) == (1.00, 5.00)


def test_dated_snapshot_resolves_to_its_family():
    assert pricing.rates("claude-haiku-4-5-20251001") == pricing.rates("claude-haiku-4-5")


def test_longest_prefix_wins():
    """claude-opus-5 and claude-opus-4-8 must not collide."""
    assert pricing.rates("claude-opus-5")[0] == 5.00
    assert pricing.rates("claude-sonnet-5")[0] == 3.00
    assert pricing.rates("claude-fable-5")[0] == 10.00


def test_unknown_model_bills_at_the_most_expensive_tier():
    """Under-charging an unknown model would let it slip past the cap."""
    unknown = pricing.rates("claude-something-unreleased")[0]
    assert unknown >= max(pricing.rates(m)[0] for m in ("claude-opus-5", "claude-haiku-4-5"))
    assert not pricing.is_known_model("claude-something-unreleased")


def test_cache_rates_are_derived_from_input_rate():
    inp, _, read, write = pricing.rates("claude-haiku-4-5")
    assert read == pytest.approx(inp * 0.10)
    assert write == pytest.approx(inp * 1.25)


def test_cache_reads_are_not_discounted_twice():
    """input_tokens is already the uncached remainder.

    The old formula did `input_tokens - cache_read_tokens`, which under-charged
    whenever the prompt cache hit — and went negative on a full cache hit.
    """
    cost = pricing.cost_usd(
        "claude-haiku-4-5",
        input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    # 1M uncached at $1.00 + 1M cache reads at $0.10
    assert cost == pytest.approx(1.10)
    assert cost > 0


def test_cost_from_usage_reads_anthropic_field_names():
    cost = pricing.cost_from_usage("claude-haiku-4-5", _usage(inp=1_000_000, out=1_000_000))
    assert cost == pytest.approx(6.00)


# ── Metering ──────────────────────────────────────────────────────────────────


def test_metered_client_records_calls_that_never_touch_state():
    """Metering happens at the client, so a call site can't opt out by forgetting."""
    from agents.analyze.node_shared import _MeteredMessages

    inner = SimpleNamespace(
        create=lambda **kw: SimpleNamespace(usage=_usage(inp=1_000_000))
    )
    messages = _MeteredMessages(inner)

    with spend.meter() as m:
        messages.create(model="claude-haiku-4-5", messages=[])
        messages.create(model="claude-haiku-4-5", messages=[])

    assert m.calls == 2
    assert m.total_usd == pytest.approx(2.00)


def test_record_outside_a_meter_is_harmless():
    assert spend.record("claude-haiku-4-5", SimpleNamespace(usage=_usage(inp=1000))) > 0


def test_record_never_raises_on_a_malformed_response():
    assert spend.record("claude-haiku-4-5", SimpleNamespace()) == 0.0
    assert spend.record("claude-haiku-4-5", object()) == 0.0


def test_concurrent_runs_meter_independently():
    """Each run executes on its own worker thread and must not bill the others."""
    totals = {}

    def run(name, calls):
        with spend.meter() as m:
            for _ in range(calls):
                spend.record("claude-haiku-4-5", SimpleNamespace(usage=_usage(inp=1_000_000)))
            totals[name] = m.total_usd

    threads = [threading.Thread(target=run, args=(n, c)) for n, c in (("a", 1), ("b", 3))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert totals["a"] == pytest.approx(1.00)
    assert totals["b"] == pytest.approx(3.00)


# ── Budget scoping ────────────────────────────────────────────────────────────


def test_guest_budget_is_keyed_on_ip_not_the_disposable_user_id():
    """POST /auth/guest mints a fresh uuid, so a user-keyed budget resets free."""
    first = budget.scope_for("guest-aaaaaaaa", "203.0.113.5")
    second = budget.scope_for("guest-bbbbbbbb", "203.0.113.5")
    assert first == second == "ip:203.0.113.5"


def test_registered_users_are_keyed_on_user_id():
    assert budget.scope_for("user-42", "203.0.113.5") == "user:user-42"


def test_guest_limit_is_tighter_than_the_user_limit():
    assert budget.limit_for("ip:203.0.113.5") < budget.limit_for("user:user-42")


# ── Budget enforcement ────────────────────────────────────────────────────────


def test_run_is_blocked_once_the_user_budget_is_spent(monkeypatch):
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1.00")
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")

    async def scenario():
        await budget.check_budget("user-42", "203.0.113.5")  # under budget
        await budget.record_spend("user:user-42", 1.50)
        with pytest.raises(Exception) as exc:
            await budget.check_budget("user-42", "203.0.113.5")
        assert exc.value.status_code == 429

    asyncio.run(scenario())


def test_new_guest_identity_cannot_reset_a_spent_budget(monkeypatch):
    monkeypatch.setenv("LLM_GUEST_DAILY_BUDGET_USD", "0.50")
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")

    async def scenario():
        await budget.record_spend(budget.scope_for("guest-aaaa", "198.51.100.9"), 0.75)
        # A brand-new guest id from the same IP must still be blocked.
        with pytest.raises(Exception) as exc:
            await budget.check_budget("guest-zzzz", "198.51.100.9")
        assert exc.value.status_code == 429
        # A different IP is unaffected.
        await budget.check_budget("guest-zzzz", "198.51.100.10")

    asyncio.run(scenario())


def test_global_cap_blocks_everyone_with_503(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "10.00")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1000")

    async def scenario():
        await budget.record_spend("user:someone-else", 12.00)
        with pytest.raises(Exception) as exc:
            await budget.check_budget("user-42", "203.0.113.5")
        assert exc.value.status_code == 503

    asyncio.run(scenario())


def test_zero_limit_disables_the_cap(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")

    async def scenario():
        await budget.record_spend("user:user-42", 9999.0)
        await budget.check_budget("user-42", "203.0.113.5")

    asyncio.run(scenario())


def test_record_spend_updates_both_global_and_scope(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "100")

    async def scenario():
        await budget.record_spend("user:user-42", 2.50)
        assert await budget.spend_today("user:user-42") == pytest.approx(2.50)
        assert await budget.spend_today(budget.GLOBAL_SCOPE) == pytest.approx(2.50)

    asyncio.run(scenario())


# ── X-Forwarded-For ───────────────────────────────────────────────────────────


def _request(xff=None, peer="192.0.2.1"):
    headers = {"X-Forwarded-For": xff} if xff else {}
    return SimpleNamespace(
        headers=SimpleNamespace(get=headers.get),
        client=SimpleNamespace(host=peer),
    )


def test_proxy_appended_address_wins_over_a_client_forgery(monkeypatch):
    """The client's own value sits left of what the proxies appended."""
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    from backend.api import auth_rate

    assert auth_rate.client_ip(_request(xff="1.2.3.4, 203.0.113.7")) == "203.0.113.7"


def test_internal_proxy_hops_are_skipped(monkeypatch):
    """Railway appends a CGNAT hop that differs per request.

    Keying on it gave every request its own bucket and silently disabled rate
    limiting in production -- 20 concurrent bad logins, zero 429s.
    """
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    from backend.api import auth_rate

    a = auth_rate.client_ip(_request(xff="203.0.113.7, 100.64.0.13"))
    b = auth_rate.client_ip(_request(xff="203.0.113.7, 100.64.0.99"))
    assert a == b == "203.0.113.7", "varying infra hop leaked into the bucket key"


def test_same_client_gets_one_bucket_across_requests(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    from backend.api import auth_rate

    seen = {
        auth_rate.client_ip(_request(xff=f"203.0.113.7, 100.64.0.{i}"))
        for i in range(1, 30)
    }
    assert seen == {"203.0.113.7"}


def test_falls_back_to_peer_when_no_forwarded_header(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    from backend.api import auth_rate

    assert auth_rate.client_ip(_request()) == "192.0.2.1"


def test_explicit_hop_count_still_honoured(monkeypatch):
    """An operator who knows their topology can pin it."""
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    from backend.api import auth_rate

    assert auth_rate.client_ip(_request(xff="1.2.3.4, 203.0.113.7")) == "203.0.113.7"


# ── Run billing ───────────────────────────────────────────────────────────────


class _SpendingGraph:
    """Stands in for a graph whose nodes make LLM calls."""

    def __init__(self, usd_per_call=1.0, fail=False):
        self.usd_per_call = usd_per_call
        self.fail = fail

    def stream(self, arg, config, stream_mode=None):
        spend.record(
            "claude-haiku-4-5",
            SimpleNamespace(usage=_usage(inp=int(self.usd_per_call * 1_000_000))),
        )
        if self.fail:
            raise RuntimeError("node blew up after spending tokens")
        return iter(())

    def get_state(self, config):
        return SimpleNamespace(values={"done": True})


async def _drain_runs():
    """Let spawned invoke tasks finish so their spend is billed."""
    tasks = [t for t in run_manager._active_tasks if not t.done()]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)


@pytest.fixture(autouse=True)
def _reset_run_manager():
    yield
    run_manager._active_tasks.clear()
    run_manager._queues.clear()
    run_manager._run_scopes.clear()
    run_manager._active_invokes = 0
    if run_manager._graph_executor is not None:
        run_manager._graph_executor.shutdown(wait=False, cancel_futures=True)
        run_manager._graph_executor = None


def test_successful_run_bills_its_scope():
    async def scenario():
        await run_manager.start_run(
            _SpendingGraph(), "run-bill", {}, "user-42", budget_scope="user:user-42"
        )
        await _drain_runs()
        return await budget.spend_today("user:user-42")

    assert asyncio.run(scenario()) == pytest.approx(1.00)


def test_failed_run_still_bills_the_tokens_it_spent():
    """A run that crashes mid-analysis has already paid Anthropic."""

    async def scenario():
        await run_manager.start_run(
            _SpendingGraph(fail=True), "run-fail", {}, "user-42", budget_scope="user:user-42"
        )
        await _drain_runs()
        return await budget.spend_today("user:user-42")

    assert asyncio.run(scenario()) == pytest.approx(1.00)


def test_create_run_endpoint_rejects_an_exhausted_budget(monkeypatch):
    """The cap has to fire in the route, not just in the budget module."""
    from backend.api.routes import runs as runs_route

    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1.00")
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "0")

    request = _request()
    request.app = SimpleNamespace(state=SimpleNamespace(graph=_SpendingGraph()))
    payload = runs_route.StartRunRequest(task="How did signups trend last week?")
    user = {"user_id": "user-42"}

    async def scenario():
        created = await runs_route.create_run(payload, request, user)
        await _drain_runs()
        assert "run_id" in created
        # That run's spend blew the budget; the next one must be refused.
        with pytest.raises(Exception) as exc:
            await runs_route.create_run(payload, request, user)
        assert exc.value.status_code == 429
        # Distinguish the budget cap from the pre-existing run rate limiter,
        # which also returns 429.
        assert "usage limit" in exc.value.detail.lower()

    asyncio.run(scenario())


def test_run_billing_falls_back_to_owner_when_scope_is_missing():
    async def scenario():
        await run_manager.start_run(_SpendingGraph(), "run-noscope", {}, "user-42")
        await _drain_runs()
        return await budget.spend_today("user:user-42")

    assert asyncio.run(scenario()) == pytest.approx(1.00)
