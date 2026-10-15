from analysis.experiment import load_experiment
from analysis.formatting import THIN_SPACE
from analysis.summary import SUMMARY_FILE, summary_markdown, write_summary


def test_classification(make_experiment) -> None:
    text = summary_markdown(load_experiment(make_experiment()))

    assert text.startswith("# Eksperyment `experiment`\n")
    assert "| Zadanie | klasyfikacja |" in text
    assert "| Architektura | 2-4-3 (27 parametrów, z biasem) |" in text
    assert "`leaky_relu(alpha=0.01)`" in text
    assert "| Inicjalizacja wag | `he` |" in text
    assert "SGD z momentum: `learning_rate` = 0,01, `momentum` = 0,9" in text
    assert "| Early stopping | nie |" in text
    assert "| Zakończenie | wszystkie epoki |" in text
    assert "| Zbiór | Strata | Dokładność |" in text
    assert "| uczący | 0,1 | 90,00% |" in text
    assert "walidacyjny" not in text
    assert "Najlepsza epoka" not in text
    assert "MSE i MAE" not in text
    assert "commit `0123456`" in text
    assert "niezatwierdzonymi" not in text


def test_regression(make_experiment) -> None:
    text = summary_markdown(load_experiment(make_experiment(task="regression", features=1)))

    assert "| Zbiór | Strata | MSE | MAE |" in text
    assert f"| testowy | 0,1 | 1{THIN_SPACE}234,5 | 12,5 |" in text
    assert "| Normalizacja wejść / celu | standaryzacja / standaryzacja |" in text
    assert "MSE i MAE w oryginalnej skali celu." in text


def test_early_stopping(make_experiment) -> None:
    text = summary_markdown(load_experiment(make_experiment(validation=True, early_stopped=True)))

    assert "| Część walidacyjna | 20% zbioru uczącego |" in text
    assert "uczący: 32, walidacyjny: 8, testowy: 30" in text
    assert "`patience` = 2, `min_delta` = 1·10⁻⁴" in text
    assert "| Zakończenie | early stopping |" in text
    assert "| Najlepsza epoka | 2 |" in text
    assert "| walidacyjny |" in text


def test_diverged(make_experiment) -> None:
    text = summary_markdown(load_experiment(make_experiment(diverged=True)))

    assert "| Zakończenie | dywergencja (epoka 4, iteracja 11) |" in text


def test_write(make_experiment, tmp_path) -> None:
    path = write_summary(load_experiment(make_experiment()), tmp_path)

    assert path == tmp_path / SUMMARY_FILE
    assert path.read_text(encoding="utf-8").endswith("\n")
