"""TEDS / TEDS-Struct — Tree-Edit-Distance-based Similarity.

Follows the formulation from IBM's PubTabNet reference implementation:
tables become labeled trees, APTED computes the edit distance, and the score is

    TEDS = 1 - distance / max(|T_pred|, |T_true|)

``structure_only=True`` (TEDS-Struct) ignores cell text, so the score reflects
structure alone rather than blending in OCR accuracy. That is the primary metric
here — the model is being evaluated on structure, and plain TEDS would let text
errors masquerade as structural ones.
"""

from __future__ import annotations

from apted import APTED, Config

from src.data.html_utils import parse_table

CELL_TAGS = ("td", "th")


class TableTree:
    """A tree node. APTED reaches children via ``Config.children``."""

    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag, colspan=1, rowspan=1, content=None, children=None):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = children if children is not None else []


def _levenshtein(a: str, b: str) -> int:
    """Iterative two-row Levenshtein.

    Inlined to avoid the unmaintained `distance` package the reference
    implementation imports.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class _TEDSConfig(Config):
    valuecls = float

    def __init__(self, structure_only: bool):
        self.structure_only = structure_only

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        """Cost of relabeling one node to the other.

        Tag or span mismatch is a full-cost edit. Text mismatch (full TEDS only)
        costs the normalized edit distance, so a near-miss is scored gently.
        """
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        if self.structure_only or node1.tag not in CELL_TAGS:
            return 0.0
        c1, c2 = node1.content or "", node2.content or ""
        if not c1 and not c2:
            return 0.0
        return _levenshtein(c1, c2) / max(len(c1), len(c2))


def _span(el, attr: str) -> int:
    try:
        return max(1, int(el.get(attr, 1)))
    except (TypeError, ValueError):
        return 1


def _build_tree(element, structure_only: bool) -> TableTree:
    if element.tag in CELL_TAGS:
        content = None if structure_only else " ".join(element.text_content().split())
        return TableTree(
            tag=element.tag,
            colspan=_span(element, "colspan"),
            rowspan=_span(element, "rowspan"),
            content=content,
        )
    node = TableTree(tag=element.tag)
    for child in element:
        if child.tag in ("tr", "td", "th", "thead", "tbody"):
            node.children.append(_build_tree(child, structure_only))
    return node


def _count_nodes(node: TableTree) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def teds_score(pred_html: str, true_html: str, structure_only: bool = True) -> float:
    """Score a prediction against ground truth. Returns a value in [0, 1].

    Unparseable or empty predictions score 0.0 — a model that emits garbage
    should not be rewarded for it.
    """
    pred_table = parse_table(pred_html)
    true_table = parse_table(true_html)
    if true_table is None:
        raise ValueError("ground-truth HTML contains no parseable <table>")
    if pred_table is None:
        return 0.0

    pred_tree = _build_tree(pred_table, structure_only)
    true_tree = _build_tree(true_table, structure_only)

    n_nodes = max(_count_nodes(pred_tree), _count_nodes(true_tree))
    if n_nodes == 0:
        return 0.0

    distance = APTED(pred_tree, true_tree, _TEDSConfig(structure_only)).compute_edit_distance()
    return max(0.0, 1.0 - float(distance) / n_nodes)


def teds_batch(
    preds: list[str], truths: list[str], structure_only: bool = True
) -> list[float]:
    """Score aligned lists. Individual failures score 0.0 rather than aborting."""
    if len(preds) != len(truths):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(truths)} truths")
    scores = []
    for pred, truth in zip(preds, truths):
        try:
            scores.append(teds_score(pred, truth, structure_only))
        except ValueError:
            scores.append(0.0)
    return scores
