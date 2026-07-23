"""PaddleOCR-VL-1.6 local inference for the table-SOTA ceiling and on-prem labels.

PaddleOCR-VL-1.6 is a 0.9B document-parsing pipeline, not an OpenAI-style chat
VLM, so it runs through its own local pipeline rather than ``vllm_client``. It
fills two roles from the plan:

1. the **table-SOTA ceiling** on public FinTabNet (OmniDocBench v1.6 Table-TEDS-S
   97.11 -- the "beats the general-VLM anchor on tables" evidence), and
2. the **on-prem ground-truth drafter** for the 20 confidential invoices, which
   cannot leave the network -- this runs locally, no API involved.

At 0.9B it runs on a Mac (MPS), a free Colab T4, or the on-prem box. All heavy
imports are deferred so this module still imports without ``paddlepaddle`` /
``paddleocr`` installed (they live in an optional requirements group, not
``requirements-base.txt``).

NOTE: the exact PaddleOCR-VL result-object shape is the one open item flagged in
the plan. ``_extract_table_html`` is written defensively against the documented
pipeline output (per-page results carrying markdown / structured blocks) and
should be confirmed on the first real run, budgeting a short debugging pass -- the
same "first GPU run is a debugging session" rule the repo already uses for
``inference.py``.
"""

from __future__ import annotations

from pathlib import Path

# Shared with mineru_client -- the defensive nested-result walk lives in one place
# now. Re-exported under the old private name so callers importing
# ``paddle_client._extract_table_html`` (e.g. the specialist notebook) still work.
from src.model.docparse_utils import extract_table_html as _extract_table_html
from src.model.registry import MODEL_PADDLEOCR_VL
from src.model.inference import Prediction
from src.model.prompts import clean_prediction


class PaddleTableReconstructor:
    """Wrap the PaddleOCR-VL pipeline behind the ``predict`` / ``predict_many``
    surface, so its predictions flow into the same eval path as every other
    candidate (``evaluate_predictions`` scores plain HTML strings)."""

    def __init__(self, model_id: str = MODEL_PADDLEOCR_VL, **pipeline_kwargs):
        self.model_id = model_id
        self._pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            # Deferred: paddle is not in requirements-base and must not be needed
            # to import this module.
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(**self._pipeline_kwargs)
        return self._pipeline

    def predict(self, image_path: str | Path, **_) -> Prediction:
        """Parse one image and return the first table as HTML.

        Extra kwargs (``mode``, ``instruction``, ...) are accepted and ignored so
        the call site stays identical to the chat backends; PaddleOCR-VL is a
        fixed document parser, not prompt-steered.
        """
        results = self._get_pipeline().predict(str(image_path))
        raw = _extract_table_html(results)
        return Prediction(uid=Path(image_path).stem, raw=raw, html=clean_prediction(raw))

    def predict_many(self, image_paths: list[str | Path], **kwargs) -> list[Prediction]:
        preds = []
        for i, path in enumerate(image_paths, 1):
            preds.append(self.predict(path, **kwargs))
            if i % 10 == 0 or i == len(image_paths):
                print(f"  predicted {i}/{len(image_paths)}")
        return preds
