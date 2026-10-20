from analysis.study import load_results, load_study, variants_table
from analysis.study_summary import study_summary_markdown


def _summary(study_dir, representative=None) -> str:
    results = load_results(load_study(study_dir))
    return study_summary_markdown(results, variants_table(results), representative or {})


def test_lists_plan_base_and_results(make_study) -> None:
    summary = _summary(make_study(missing=("runs/tanh__a/seed-2",)))

    assert summary.startswith("# Badanie `study`\n\nSynthetic study\n")
    assert "| `activation` | `model.bias` | `sigmoid`, `tanh` |" in summary
    assert "| Ziarna | 1, 2, 3 |" in summary
    assert "| Liczba runów | 12 (4 x 3) |" in summary
    assert "| `task` | `classification` |" in summary
    assert "Bez artefaktów" in summary
    # Mean and sample deviation of 90,1%, 90,2%, 90,3%.
    assert "| `sigmoid` | `a` | 3/3 | 0 | 0 | 90,20% ± 0,10 | 90,20% ± 0,10 |" in summary
    assert "| `tanh` | `a` | 2/3 | 0 | 1 |" in summary
    assert "Runy reprezentatywne" not in summary


def test_marks_keys_set_by_axes(make_study) -> None:
    study_dir = make_study(axes={"bias": ["on", "off"]})
    with (study_dir / "study.toml").open("a", encoding="utf-8") as file:
        file.write("\n[base.model]\nbias = true\n")

    summary = _summary(study_dir)

    assert "| `training.loss` | `mse` |" in summary
    assert "| `model.bias` | zmieniane przez oś `bias` |" in summary


def test_regression_errors_without_percent(make_study) -> None:
    summary = _summary(make_study(task="regression", axes={"a": ["x"]}, seeds=(1,)))

    assert "MSE (testowy)" in summary
    assert "MAE (testowy)" in summary
    assert "%" not in summary.split("## Wyniki")[1]


def test_regression_errors_as_median_and_quartiles(make_study) -> None:
    summary = _summary(make_study(task="regression", axes={"a": ["x"]}))

    results = summary.split("## Wyniki")[1]
    assert chr(0xB1) not in results
    assert chr(0x2013) in results
    assert "Mediana (w nawiasie kwartyle Q1–Q3)" in results


def test_labels_in_results_table(make_study) -> None:
    report = '[report]\nlabels = { activation = "Aktywacja", tanh = "tangens" }'
    summary = _summary(make_study(report=report))

    results = summary.split("## Wyniki")[1]
    assert "| Aktywacja | `dataset` |" in results
    assert "| tangens | `a` |" in results


def test_names_compared_split(make_study) -> None:
    default = _summary(make_study())
    assert "| Porównanie wariantów | zbiór testowy (metryka testowa) |" in default

    study_dir = make_study(validation=True, report='[report]\nsplit = "validation"')
    summary = _summary(study_dir, {"sigmoid__a": study_dir / "runs/sigmoid__a/seed-2"})

    assert "| Porównanie wariantów | zbiór walidacyjny (metryka walidacyjna) |" in summary
    assert "Dokładność (walidacyjny)" in summary
    assert "run, którego metryka walidacyjna jest najbliższa medianie" in summary
