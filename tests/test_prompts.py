"""Tests for prompt construction and model-output cleaning.

``clean_prediction`` sits between raw generation and every score, so a bug here
shows up as a bad TEDS number rather than as an error. Base VLMs reliably wrap
output in markdown fences and add commentary despite being told not to; each
case below is a shape real models actually emit.
"""

import pytest

from src.model.prompts import (
    DEFAULT_INVOICE_SCHEMA,
    FULL_INSTRUCTION,
    SCHEMA_INSTRUCTION,
    STRUCTURE_INSTRUCTION,
    append_bbox_rule,
    build_messages,
    build_schema_instruction,
    build_training_example,
    clean_prediction,
    resolve_instruction,
)

TABLE = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'


class TestCleanPrediction:
    def test_bare_html_passes_through(self):
        assert clean_prediction(TABLE) == TABLE

    def test_strips_markdown_fence(self):
        assert clean_prediction(f"```html\n{TABLE}\n```") == TABLE

    def test_strips_unlabelled_fence(self):
        assert clean_prediction(f"```\n{TABLE}\n```") == TABLE

    def test_strips_leading_prose(self):
        raw = f"Here is the reconstructed table:\n\n{TABLE}"
        assert clean_prediction(raw) == TABLE

    def test_strips_trailing_commentary(self):
        raw = f"{TABLE}\n\nNote that the first cell spans two rows."
        assert clean_prediction(raw) == TABLE

    def test_strips_prose_on_both_sides(self):
        raw = f"Sure! Here it is:\n```html\n{TABLE}\n```\nLet me know if you need changes."
        assert clean_prediction(raw) == TABLE

    def test_no_table_returns_empty(self):
        """Scored as 0 by the caller rather than guessed at."""
        assert clean_prediction("I cannot read this image.") == ""

    def test_empty_input_returns_empty(self):
        assert clean_prediction("") == ""
        assert clean_prediction("   \n  ") == ""

    def test_unterminated_table_is_kept(self):
        """Truncation at max_new_tokens is common; lxml recovers the partial
        tree, so keeping it scores better than discarding it."""
        truncated = '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
        assert clean_prediction(truncated).startswith("<table")

    def test_uppercase_tags_are_found(self):
        assert clean_prediction("<TABLE><TR><TD>x</TD></TR></TABLE>").startswith("<TABLE")

    def test_prefers_the_fenced_block_containing_a_table(self):
        raw = "```python\nprint('hi')\n```\nand the table:\n```html\n" + TABLE + "\n```"
        assert clean_prediction(raw) == TABLE


class TestInstructionResolution:
    def test_defaults_by_mode(self):
        assert resolve_instruction("structure") == STRUCTURE_INSTRUCTION
        assert resolve_instruction("full") == FULL_INSTRUCTION

    def test_override_wins(self):
        assert resolve_instruction("structure", "CUSTOM") == "CUSTOM"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            resolve_instruction("nonsense")

    def test_override_bypasses_mode_validation(self):
        """Prompt iteration should not require inventing a mode name."""
        assert resolve_instruction("nonsense", "CUSTOM") == "CUSTOM"

    def test_structure_prompt_forbids_cell_text(self):
        """The training target has no text; the prompt must ask for none."""
        assert "no cell" in STRUCTURE_INSTRUCTION.lower()


class TestMessageShape:
    def test_user_turn_carries_image_then_text(self):
        msgs = build_messages("img.png", "structure")
        assert msgs[0]["role"] == "user"
        kinds = [c["type"] for c in msgs[0]["content"]]
        assert kinds == ["image", "text"]

    def test_override_reaches_the_message(self):
        """The Day 2 path: iterate prompts from the notebook without editing
        prompts.py, which an already-imported module would ignore anyway."""
        msgs = build_messages("img.png", instruction="CUSTOM PROMPT")
        assert msgs[0]["content"][1]["text"] == "CUSTOM PROMPT"

    def test_training_example_appends_assistant_turn(self):
        ex = build_training_example("img.png", TABLE)
        assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
        assert ex["messages"][1]["content"][0]["text"] == TABLE


class TestSchemaInstruction:
    """The schema path must generalise: teach principles, not one fixed layout."""

    def test_default_is_document_agnostic(self):
        """The default (no schema arg) must NOT bake in one sample's column names
        -- that overfits and only works on documents matching that sample."""
        default = build_schema_instruction()
        assert default == SCHEMA_INSTRUCTION
        for hardcoded in ("Currency", "Figure100", "Figures", "Debit", "Credit"):
            assert hardcoded not in default

    def test_default_teaches_the_reconstruction_principles(self):
        low = build_schema_instruction().lower()
        assert "logical" in low
        assert "blank" in low          # sparse cells preserved
        assert "variable" in low       # variable column count
        assert "<th>" in low           # hierarchical headers

    def test_schema_hint_is_included_but_framed_as_a_hint(self):
        """A supplied header appears verbatim, but as a clue to verify -- never as
        a structure to force."""
        with_hint = build_schema_instruction(DEFAULT_INVOICE_SCHEMA)
        assert "Figure100" in with_hint
        assert "hint" in with_hint.lower()
        assert "verify" in with_hint.lower()

    def test_ignore_rule_covers_handwriting_and_qr(self):
        low = build_schema_instruction().lower()
        assert "handwritten" in low
        assert "qr" in low
        assert "barcode" in low

    def test_warns_about_cropped_clipped_borders(self):
        """Input is a YOLOX table crop whose outer border may be clipped -- the
        model must not read a missing edge as a dropped/merged column."""
        low = build_schema_instruction().lower()
        assert "crop" in low
        assert "clip" in low
        assert "border" in low

    def test_bbox_clamped_to_image(self):
        low = build_schema_instruction(with_bbox=True).lower()
        assert "inside the image" in low or "image border" in low

    def test_bbox_off_by_default(self):
        assert "data-bbox" not in build_schema_instruction()

    def test_with_bbox_requests_per_cell_boxes(self):
        instr = build_schema_instruction(with_bbox=True)
        assert "data-bbox" in instr
        low = instr.lower()
        assert "x1,y1,x2,y2" in low
        assert "one box per logical cell" in low

    def test_append_bbox_rule_extends_any_instruction(self):
        extended = append_bbox_rule(FULL_INSTRUCTION)
        assert extended.startswith(FULL_INSTRUCTION)
        assert "data-bbox" in extended
