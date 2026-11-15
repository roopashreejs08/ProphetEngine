"""Smoke tests: every dashboard page renders without raising."""

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.config import REPO_ROOT

PAGES = [
    ("overview", "render"),
    ("live", "render"),
    ("predict", "render_single"),
    ("predict", "render_batch"),
    ("performance", "render"),
    ("explain", "render"),
    ("data_explorer", "render"),
    ("about", "render"),
]


def _page_script(module, function):
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"from dashboard.views import {module}\n"
        f"{module}.{function}()\n"
    )


@pytest.mark.parametrize(("module", "function"), PAGES)
def test_page_renders(module, function):
    at = AppTest.from_string(_page_script(module, function), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_entrypoint_renders():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
