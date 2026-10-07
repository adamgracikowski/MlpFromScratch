import matplotlib
import numpy
import polars

import analysis
from analysis.cli import build_parser


def test_dependencies_import() -> None:
    assert numpy.__version__
    assert matplotlib.__version__
    assert polars.__version__


def test_parser_builds() -> None:
    assert analysis.__version__
    assert build_parser().prog == "analysis"
