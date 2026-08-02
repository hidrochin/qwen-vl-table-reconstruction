"""Tests for the reading pass (src/ocr/read.py) -- the pure geometry only.

The model call and PIL are injected/deferred, so the orchestrator is not
exercised here; tiling, parsing/offsetting, cross-read consensus, the fragment
block, and OcrWord-compatibility are, since those are what make the pass
trustworthy by construction.
"""

from src.ocr.layout import to_grid_text
from src.ocr.read import (
    Fragment,
    _iou,
    fragments_json_schema,
    merge_fragments,
    parse_fragments,
    plan_tiles,
    render_fragment_block,
)


class TestPlanTiles:
    def test_small_image_is_one_tile(self):
        assert plan_tiles(800, 600, tile=1024) == [(0, 0, 800, 600)]

    def test_large_image_tiles_with_overlap_and_edge_flush(self):
        tiles = plan_tiles(2000, 1000, tile=1024, overlap=0.2)
        assert len(tiles) > 1
        # first tile at the origin, last tile flush to the right edge
        assert tiles[0][:2] == (0, 0)
        assert max(t[2] for t in tiles) == 2000
        assert max(t[3] for t in tiles) == 1000

    def test_degenerate_size(self):
        assert plan_tiles(0, 0) == []


class TestParseFragments:
    def test_parses_and_offsets_and_converts_to_xywh(self):
        raw = '{"fragments":[{"t":"Return","b":[10,20,60,35]},{"t":"3%","b":[100,20,130,35]}]}'
        frs = parse_fragments(raw, origin=(5, 5), page=2)
        assert [f.text for f in frs] == ["Return", "3%"]
        # box offset by origin and stored as (x, y, w, h)
        assert frs[0].bbox == (15.0, 25.0, 50.0, 15.0)
        assert all(f.page == 2 for f in frs)

    def test_drops_empty_and_degenerate(self):
        raw = '{"fragments":[{"t":"","b":[0,0,10,10]},{"t":"x","b":[5,5,5,5]},{"t":"ok","b":[0,0,4,4]}]}'
        frs = parse_fragments(raw)
        assert [f.text for f in frs] == ["ok"]

    def test_tolerates_fences_and_thinking(self):
        raw = '<think>reading</think>```json\n{"fragments":[{"t":"A","b":[0,0,5,5]}]}\n```'
        assert [f.text for f in parse_fragments(raw)] == ["A"]

    def test_garbage_is_empty(self):
        assert parse_fragments("no json") == []
        assert parse_fragments("") == []


class TestMergeAndConsensus:
    def test_overlapping_reads_dedup_to_one_with_majority_text(self):
        # three reads of the same physical fragment; two agree on "123"
        group = [
            Fragment(0, "123", (10, 10, 30, 15)),
            Fragment(0, "128", (11, 11, 30, 15)),
            Fragment(0, "123", (10, 10, 31, 15)),
        ]
        merged = merge_fragments(group)
        assert len(merged) == 1
        f = merged[0]
        assert f.text == "123"
        assert f.contested is True
        assert "128" in f.alternates
        assert 0.6 < f.conf < 0.7  # 2 of 3 agree

    def test_distinct_fragments_are_kept_separate_and_reordered(self):
        frs = [
            Fragment(0, "B", (100, 0, 20, 10)),
            Fragment(0, "A", (0, 0, 20, 10)),
        ]
        merged = merge_fragments(frs)
        assert [f.text for f in merged] == ["A", "B"]  # reading order
        assert [f.id for f in merged] == [0, 1]  # stable ids assigned

    def test_iou(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


class TestFragmentInterop:
    def test_fragment_is_ocrword_compatible(self):
        frs = [Fragment(0, "Return", (0, 40, 50, 12)), Fragment(1, "3%", (120, 40, 30, 12))]
        # layout.py consumes it unchanged
        assert "Return" in to_grid_text(frs)
        f = frs[0]
        assert (f.x, f.y, f.w, f.h) == (0, 40, 50, 12)
        assert f.box_xyxy == (0, 40, 50, 52)

    def test_fragment_block_lists_ids(self):
        block = render_fragment_block([Fragment(7, "D", (200, 40, 12, 12), alternates=("O",))])
        assert block.startswith("7:")
        assert '"D"' in block and "alt=['O']" in block


class TestGuidedSchema:
    def test_structure_blind_contract(self):
        item = fragments_json_schema()["properties"]["fragments"]["items"]
        assert set(item["required"]) == {"t", "b"}
        # nothing structural is allowed in a reading-pass fragment
        assert "r" not in item["properties"] and "c" not in item["properties"]
