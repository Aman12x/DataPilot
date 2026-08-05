import os

import pytest

from tools.db_tools import DBConnection


pytestmark = pytest.mark.integration


def _pg_config() -> dict[str, object]:
    if os.getenv("DATAPILOT_POSTGRES_INTEGRATION") != "1":
        pytest.skip("DATAPILOT_POSTGRES_INTEGRATION=1 is not configured")
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "datapilot_test"),
        "user": os.getenv("PGUSER", "datapilot"),
        "password": os.getenv("PGPASSWORD", "datapilot"),
    }


def test_postgres_query_schema_and_readonly_guards():
    psycopg2 = pytest.importorskip("psycopg2")
    cfg = _pg_config()

    admin = psycopg2.connect(**cfg)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS datapilot_ci_metrics")
            cur.execute("""
                CREATE TABLE datapilot_ci_metrics (
                    day date PRIMARY KEY,
                    metric integer NOT NULL,
                    segment text NOT NULL
                )
            """)
            cur.execute("""
                INSERT INTO datapilot_ci_metrics(day, metric, segment)
                VALUES ('2026-01-01', 10, 'control'),
                       ('2026-01-02', 12, 'treatment')
            """)
    finally:
        admin.close()

    db = DBConnection("postgres", **cfg)
    df = db.query("""
        /* leading comments are allowed */
        SELECT segment, SUM(metric) AS total_metric
        FROM datapilot_ci_metrics
        GROUP BY segment
        ORDER BY segment
    """)

    assert df["segment"].tolist() == ["control", "treatment"]
    assert df["total_metric"].tolist() == [10, 12]

    with pytest.raises(ValueError):
        db.query("CREATE TABLE should_not_run(id integer)")

    schema = db.inspect_schema()
    assert "TABLE: datapilot_ci_metrics" in schema
    assert "metric" in schema


def test_postgres_checkpointer_roundtrips_dataframe_state():
    """The real fix for split-brain storage: PostgresSaver + SafeCheckpointSerde
    must round-trip a checkpoint whose channel values include a DataFrame —
    the shape every gate interrupt persists."""
    pytest.importorskip("langgraph.checkpoint.postgres")
    cfg = _pg_config()
    url = (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )

    import pandas as pd
    from langgraph.checkpoint.base import empty_checkpoint

    from backend.api.main import _make_postgres_checkpointer

    saver, pool = _make_postgres_checkpointer(url)
    try:
        df = pd.DataFrame({"variant": ["control", "treatment"], "dau": [0.65, 0.64]})
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"query_result": df, "task": "roundtrip"}
        # A non-primitive channel value is stored out-of-line in checkpoint_blobs,
        # and both halves of that round-trip are keyed on the channel version:
        # `put` writes a blob only for channels in `new_versions`, and the read
        # joins blobs on `checkpoint->'channel_versions'`. Set both, to the same
        # versions, or the DataFrame is dropped on write and absent on read.
        versions = {"query_result": "1", "task": "1"}
        checkpoint["channel_versions"] = versions
        config = {"configurable": {"thread_id": "ci-roundtrip", "checkpoint_ns": ""}}

        saved = saver.put(config, checkpoint, {"source": "test", "step": 1}, versions)
        got = saver.get_tuple(saved)
        assert got is not None
        restored = got.checkpoint["channel_values"]
        assert restored["task"] == "roundtrip"
        pd.testing.assert_frame_equal(restored["query_result"], df)
    finally:
        pool.close()
