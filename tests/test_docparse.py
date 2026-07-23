"""The specialist clients (PaddleOCR-VL, MinerU) share one defensive output
parser because neither result-object shape is confirmed yet. These tests pin the
walk over nested dict/list/attr output and the predict surface, with the frameworks
mocked out so they run on the Mac.
"""

from src.model.docparse_utils import extract_table_html
from src.model.mineru_client import MinerUTableReconstructor
from src.model.paddle_client import PaddleTableReconstructor

TABLE = "<table><tr><td>a</td><td>b</td></tr></table>"


class _Page:
    """Stand-in for a per-page result object exposing ``.markdown``."""

    def __init__(self, markdown):
        self.markdown = markdown


class _FakePipeline:
    def __init__(self, result):
        self._result = result

    def predict(self, _path):
        return self._result


class TestExtractTableHtml:
    def test_finds_table_in_nested_dict(self):
        result = {"pages": [{"content": f"prose\n{TABLE}\nmore"}]}
        assert extract_table_html(result) == TABLE

    def test_finds_table_via_object_attr(self):
        assert extract_table_html([_Page(f"# heading\n{TABLE}")]) == TABLE

    def test_returns_empty_when_no_table(self):
        assert extract_table_html({"pages": ["no table here"]}) == ""

    def test_handles_none(self):
        assert extract_table_html(None) == ""


class TestSpecialistPredict:
    def test_paddle_predict_extracts_and_cleans(self):
        recon = PaddleTableReconstructor()
        recon._pipeline = _FakePipeline([_Page(f"```html\n{TABLE}\n```")])
        pred = recon.predict("data/invoices/inv_007.png")
        assert pred.uid == "inv_007"
        assert pred.html == TABLE  # markdown fence stripped by clean_prediction

    def test_mineru_predict_extracts_and_cleans(self):
        recon = MinerUTableReconstructor()
        recon._parser = _FakePipeline({"markdown": f"text {TABLE} text"})
        pred = recon.predict("data/invoices/inv_009.jpg")
        assert pred.uid == "inv_009" and pred.html == TABLE

    def test_empty_result_yields_empty_html(self):
        recon = PaddleTableReconstructor()
        recon._pipeline = _FakePipeline({"pages": ["nothing"]})
        assert recon.predict("x.png").html == ""
