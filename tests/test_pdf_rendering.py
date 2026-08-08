"""PDF report rendering — markdown must not reach the page as literal text.

The reported symptom was a report full of stray `**`, headings rendered as
bullets, bold bleeding across the wrong words, and `<!-- details -->` printed
verbatim. All of it traced to one character: `*` inside the bullet-detection
character class, which made `**Bottom Line**` look like a `*` bullet.

These drive `_render_markdown` against a recorder rather than a real FPDF, so
they assert what was actually drawn instead of inspecting compressed PDF bytes.
"""
import pytest

from backend.api import pdf as pdf_mod


class _Recorder:
    """Minimal stand-in for the FPDF surface `_render_markdown` draws on."""

    def __init__(self):
        self.spans: list[tuple[str, str]] = []   # (style, text)
        self.cells: list[str] = []
        self.l_margin = 20.0
        self._style = ""
        self._y = 40.0

    # font / colour / geometry — recorded only where it matters
    def set_font(self, _family, style="", _size=10):
        self._style = style

    def set_text_color(self, *_a):        pass
    def set_draw_color(self, *_a):        pass
    def set_line_width(self, *_a):        pass
    def set_left_margin(self, m):         self.l_margin = m
    def set_x(self, _x):                  pass
    def set_y(self, y):                   self._y = y
    def get_y(self):                      return self._y
    def line(self, *_a):                  pass

    def ln(self, h=0):
        self._y += h or 2

    def cell(self, _w, _h, txt="", **_kw):
        if txt:
            self.cells.append(txt)

    def write(self, _h, txt=""):
        if txt:
            self.spans.append((self._style, txt))

    # helpers
    def text(self) -> str:
        return "".join(t for _s, t in self.spans) + " " + " ".join(self.cells)

    def bold_text(self) -> str:
        return " ".join(t for s, t in self.spans if "B" in s)


def _render(md: str) -> _Recorder:
    rec = _Recorder()
    pdf_mod._render_markdown(rec, md)
    return rec


def test_bold_only_line_becomes_a_heading_not_a_bullet():
    """`**Bottom Line**` is a heading. It used to match the bullet class on `*`.

    The strip that followed removed the *opening* `**`, which both stranded the
    closing pair as literal text and shifted every later delimiter on the line by
    one, inverting bold and plain for the rest of the paragraph.
    """
    rec = _render("**Bottom Line**\nSome body text here.")
    assert "Bottom Line" in " ".join(rec.cells)
    assert "**" not in rec.text()
    # Rendered via cell() as a heading, not written as a bullet body.
    assert "Bottom Line" not in "".join(t for _s, t in rec.spans)


def test_bullet_with_a_bold_label_keeps_both_marker_and_bold():
    """`- **Run ID:** value` lost its `**` to the same greedy strip."""
    rec = _render("- **Run ID:** abc-123")
    assert "**" not in rec.text()
    assert "Run ID:" in rec.bold_text()
    assert "abc-123" in rec.text()


def test_bold_spans_pair_correctly_across_a_mixed_line():
    body = "Power **users show 85.0%** while new **users trail at 52.6%** overall."
    rec = _render(body)
    bold = rec.bold_text()
    assert "users show 85.0%" in bold
    assert "users trail at 52.6%" in bold
    assert "Power" not in bold          # leading plain text must stay plain
    assert "**" not in rec.text()


def test_html_comments_never_reach_the_page():
    rec = _render("Before\n<!-- details -->\nAfter")
    assert "details" not in rec.text()
    assert "<!--" not in rec.text()


def test_a_marker_with_no_content_is_dropped():
    """A bare `-` line printed a lone bullet in the middle of the report."""
    rec = _render("- real point\n-\n\n- another point")
    assert "real point" in rec.text()
    assert "another point" in rec.text()
    assert rec.cells.count("-") == 2   # exactly the two real bullets


def test_blockquote_renders_as_prose_without_its_marker():
    rec = _render("> **Auto-corrected:** sample size missing.")
    assert ">" not in rec.text()
    assert "Auto-corrected:" in rec.bold_text()
    assert "sample size missing." in rec.text()


def test_inline_code_backticks_are_stripped():
    rec = _render("- **Run ID:** `e64a2963-8790-4cee`")
    assert "`" not in rec.text()
    assert "e64a2963-8790-4cee" in rec.text()


def test_bullet_wrapping_is_hanging_indented():
    """Continuation lines used to wrap flush to the page margin.

    `write()` wraps to the *margin*, not the cursor, so setting only `set_x`
    indented the first line and left every wrapped line under it hanging off to
    the left. The margin has to move, and be restored afterwards.
    """
    rec = _Recorder()
    before = rec.l_margin
    pdf_mod._render_markdown(rec, "- a bullet whose text is long enough to wrap")
    assert rec.l_margin == before, "left margin leaked past the bullet"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("risk — the biggest one", "risk, the biggest one"),   # aside -> comma
        ("retention 7 — 10 percent", "retention 7 - 10 percent"),  # range kept
        ("7-day and opt-out", "7-day and opt-out"),            # hyphens intact
    ],
)
def test_spaced_dashes_become_commas_without_touching_ranges_or_hyphens(raw, expected):
    assert pdf_mod._clean(raw) == expected


def test_build_pdf_produces_a_pdf_for_a_report_with_every_construct():
    out = pdf_mod.build_pdf(
        task="segment only android users",
        narrative=(
            "**Bottom Line**\n"
            "**3 segments** analyzed — the gap is large.\n\n"
            "<!-- details -->\n"
            "- **Run ID:** `abc`\n"
            "-\n"
            "> **Auto-corrected:** missing n.\n"
        ),
        recommendation="Redesign onboarding. " * 20,   # long enough to wrap the callout
        metric="day-7 retention",
        cost_usd=0.0412,
    )
    assert out.startswith(b"%PDF")
    assert len(out) > 1000
