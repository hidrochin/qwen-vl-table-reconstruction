"""Side-by-side comparison renderer for the customer demo.

Produces one self-contained HTML file -- images inlined as base64, no external
assets -- so it opens offline from a laptop. Nothing runs live at the demo; a
free-tier GPU picking demo day to be slow is an avoidable risk.

Spanning cells are highlighted in both prediction and ground truth. Merged-cell
recovery is the hard part of the task and the reason simpler tools failed, so it
should be visible at a glance rather than buried in a metric.
"""

from __future__ import annotations

import base64
import html as html_escape
from dataclasses import dataclass
from pathlib import Path

from src.data.html_utils import extract_cells

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem;
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #fbfbfa; color: #1a1a1a;
}
h1 { font-size: 1.5rem; margin: 0 0 .35rem; }
.sub { color: #666; margin-bottom: 1.5rem; }
.summary {
  background: #fff; border: 1px solid #e3e3e0; border-radius: 10px;
  padding: 1.1rem 1.3rem; margin-bottom: 2rem;
}
.summary table { border-collapse: collapse; width: auto; }
.summary th, .summary td {
  text-align: left; padding: .4rem 1.4rem .4rem 0;
  border-bottom: 1px solid #eee; font-variant-numeric: tabular-nums;
}
.summary th { font-weight: 600; color: #555; }
.case {
  background: #fff; border: 1px solid #e3e3e0; border-radius: 10px;
  padding: 1.1rem 1.3rem; margin-bottom: 1.5rem;
}
.case-head {
  display: flex; align-items: center; gap: .6rem;
  flex-wrap: wrap; margin-bottom: .9rem;
}
.uid { font-weight: 600; }
.badge {
  font-size: .75rem; padding: .17rem .55rem; border-radius: 99px;
  border: 1px solid transparent; font-weight: 600; white-space: nowrap;
}
.b-good { background: #e6f4ea; color: #17632c; }
.b-mid  { background: #fdf3d8; color: #7a5b06; }
.b-bad  { background: #fce8e6; color: #96231a; }
.b-info { background: #eef1f5; color: #44506b; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
.pane { min-width: 0; }
.pane h3 {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
  color: #777; margin: 0 0 .5rem;
}
.viewport {
  border: 1px solid #e8e8e5; border-radius: 6px; background: #fafaf9;
  padding: .6rem; overflow: auto; max-height: 420px;
}
.viewport img { max-width: 100%; height: auto; display: block; }
.viewport table { border-collapse: collapse; font-size: .78rem; }
.viewport table td, .viewport table th {
  border: 1px solid #c8c8c4; padding: .2rem .4rem;
  vertical-align: top; text-align: left;
}
.viewport table th { background: #f0f0ee; font-weight: 600; }
.span-cell { background: #dbeafe !important; outline: 1.5px solid #3b82f6; }
.empty-note { color: #999; font-style: italic; font-size: .8rem; }
@media (prefers-color-scheme: dark) {
  body { background: #17171a; color: #e8e8e6; }
  .summary, .case { background: #1f1f23; border-color: #33333a; }
  .summary th, .summary td { border-bottom-color: #2c2c33; }
  .summary th, .sub, .pane h3 { color: #9a9aa2; }
  .viewport { background: #191a1d; border-color: #33333a; }
  .viewport table td, .viewport table th { border-color: #44444d; }
  .viewport table th { background: #26262c; }
  .b-good { background: #123322; color: #7fe0a3; }
  .b-mid  { background: #33290d; color: #f0cd77; }
  .b-bad  { background: #3a1714; color: #f5a19a; }
  .b-info { background: #24272f; color: #a8b4cd; }
  .span-cell { background: #1e3a5f !important; outline-color: #60a5fa; }
}
"""


@dataclass
class Case:
    """One row of the comparison: image, prediction, truth, score."""

    uid: str
    image_path: str
    pred_html: str
    true_html: str
    score: float
    difficulty: str = ""
    n_spanning: int = 0


def _b64_image(path: str | Path) -> str:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def _highlight_spans(table_html: str) -> str:
    """Tag spanning cells so they stand out visually.

    Rebuilt from the parsed grid rather than regexed, so malformed model output
    degrades to a plain render instead of corrupting the page.
    """
    cells = extract_cells(table_html)
    if not cells:
        return '<p class="empty-note">no parseable table</p>'

    by_row: dict[int, list] = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)

    parts = ["<table>"]
    for row in sorted(by_row):
        parts.append("<tr>")
        for c in sorted(by_row[row], key=lambda x: x.col):
            attrs = ""
            if c.rowspan > 1:
                attrs += f' rowspan="{c.rowspan}"'
            if c.colspan > 1:
                attrs += f' colspan="{c.colspan}"'
            cls = ' class="span-cell"' if c.is_spanning else ""
            text = html_escape.escape(c.text) or "&nbsp;"
            parts.append(f"<{c.tag}{attrs}{cls}>{text}</{c.tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _score_class(score: float) -> str:
    return "b-good" if score >= 0.90 else "b-mid" if score >= 0.70 else "b-bad"


def _summary_block(cases: list[Case], subtitle: str) -> str:
    if not cases:
        return ""
    by_bin: dict[str, list[float]] = {}
    for c in cases:
        by_bin.setdefault(c.difficulty or "unbinned", []).append(c.score)

    rows = "".join(
        f"<tr><td>{html_escape.escape(b)}</td><td>{len(v)}</td>"
        f"<td>{sum(v)/len(v):.3f}</td></tr>"
        for b, v in sorted(by_bin.items())
    )
    overall = sum(c.score for c in cases) / len(cases)
    return f"""<div class="summary">
  <table>
    <tr><th>Difficulty</th><th>Tables</th><th>Mean TEDS-Struct</th></tr>
    {rows}
    <tr><th>All</th><th>{len(cases)}</th><th>{overall:.3f}</th></tr>
  </table>
  <p class="sub" style="margin:.8rem 0 0">{html_escape.escape(subtitle)}</p>
</div>"""


def render_comparison(
    cases: list[Case],
    out_path: str | Path,
    title: str = "Table Structure Reconstruction",
    subtitle: str = "Metric is TEDS-Struct: structure only, text ignored.",
) -> Path:
    """Write the self-contained comparison page. Returns the path written."""
    blocks = []
    for case in cases:
        badges = [f'<span class="badge {_score_class(case.score)}">TEDS-Struct {case.score:.3f}</span>']
        if case.difficulty:
            badges.append(f'<span class="badge b-info">{html_escape.escape(case.difficulty)}</span>')
        if case.n_spanning:
            badges.append(f'<span class="badge b-info">{case.n_spanning} merged cells</span>')

        blocks.append(f"""<div class="case">
  <div class="case-head">
    <span class="uid">{html_escape.escape(case.uid)}</span>
    {''.join(badges)}
  </div>
  <div class="grid">
    <div class="pane"><h3>Input image</h3>
      <div class="viewport"><img src="{_b64_image(case.image_path)}" alt="table image"></div></div>
    <div class="pane"><h3>Model prediction</h3>
      <div class="viewport">{_highlight_spans(case.pred_html)}</div></div>
    <div class="pane"><h3>Ground truth</h3>
      <div class="viewport">{_highlight_spans(case.true_html)}</div></div>
  </div>
</div>""")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_escape.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_escape.escape(title)}</h1>
<p class="sub">Merged cells highlighted in blue. {len(cases)} hard tables.</p>
{_summary_block(cases, subtitle)}
{''.join(blocks)}
</body></html>"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
