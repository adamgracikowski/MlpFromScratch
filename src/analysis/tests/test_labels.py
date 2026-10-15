import pytest

from analysis.labels import parameter_count, plural, train_subset


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "a"),
        (2, "b"),
        (4, "b"),
        (5, "c"),
        (12, "c"),
        (14, "c"),
        (22, "b"),
        (111, "c"),
        (1153, "b"),
    ],
)
def test_plural(count: int, expected: str) -> None:
    assert plural(count, "a", "b", "c") == expected


def test_counts() -> None:
    assert parameter_count(371, bias=True) == "371 parametrów, z biasem"
    assert parameter_count(1153, bias=False) == "1153 parametry, bez biasu"
    assert train_subset(200) == "200 próbek"
    assert train_subset(22) == "22 próbki"
