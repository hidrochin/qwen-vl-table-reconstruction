"""MinerU2.5-Pro local inference -- best open Table-TEDS specialist + stage-1.

MinerU2.5-Pro (``opendatalab/MinerU2.5-Pro-2605-1.2B``) is a 1.2B decoupled
document-parsing VLM: global layout analysis on a downsampled image, then
native-resolution recognition of text / formulas / tables. It has the best
Table-TEDS among open specialists and a non-AGPL licence, so it fills two roles:

1. the **table-structure ceiling** on public FinTabNet (Track 1), and
2. the on-prem **stage-1 OCR + geometry** and ground-truth drafter for the 20
   confidential invoices (Track 2) -- runs locally, no API involved.

Like ``paddle_client`` it wraps the pipeline behind ``predict`` / ``predict_many``
so its output flows into the same eval path as every other candidate. All heavy
imports are deferred so this module still imports without ``mineru`` installed
(it lives in ``requirements-onprem.txt``, not ``requirements-base``).

NOTE: MinerU's Python entry point is the open item to confirm on the first real
run -- ``_run`` is written against the documented package surface and isolated so
that pass is a one-liner to adjust, the same "first run is a debugging session"
rule the repo uses for every serving path. MinerU2.5-Pro is itself vLLM-servable,
so an alternative is to point ``vllm_client`` at a served MinerU endpoint.
"""

from __future__ import annotations

from pathlib import Path

from src.model.docparse_utils import extract_table_html
from src.model.inference import Prediction
from src.model.prompts import clean_prediction
from src.model.registry import MODEL_MINERU


class MinerUTableReconstructor:
    """Wrap MinerU2.5-Pro behind the ``predict`` / ``predict_many`` surface."""

    def __init__(self, model_id: str = MODEL_MINERU, backend: str = "vlm", **pipeline_kwargs):
        self.model_id = model_id
        self.backend = backend
        self._pipeline_kwargs = pipeline_kwargs
        self._parser = None

    def _get_parser(self):
        if self._parser is None:
            # Deferred: mineru is not in requirements-base and must not be needed
            # to import this module.
            from mineru.backend.vlm.vlm_analyze import ModelSingleton

            self._parser = ModelSingleton().get_model(self.model_id, **self._pipeline_kwargs)
        return self._parser

    def _run(self, image_path: str | Path):
        """Parse one image, returning MinerU's raw result for ``extract_table_html``.

        Confirm this call against the installed ``mineru`` on the first run; the
        rest of the class does not care about the exact shape.
        """
        return self._get_parser().predict(str(image_path))

    def predict(self, image_path: str | Path, **_) -> Prediction:
        """Parse one image and return the first table as HTML.

        Extra kwargs (``mode``, ``instruction``, ...) are accepted and ignored so
        the call site stays identical to the chat backends; MinerU is a fixed
        document parser, not prompt-steered.
        """
        raw = extract_table_html(self._run(image_path))
        return Prediction(uid=Path(image_path).stem, raw=raw, html=clean_prediction(raw))

    def predict_many(self, image_paths: list[str | Path], **kwargs) -> list[Prediction]:
        preds = []
        for i, path in enumerate(image_paths, 1):
            preds.append(self.predict(path, **kwargs))
            if i % 10 == 0 or i == len(image_paths):
                print(f"  predicted {i}/{len(image_paths)}")
        return preds
