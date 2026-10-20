import pytest

from analysis.study import Axis, Roles, StudyError, load_study


def test_loads_axes_seeds_and_index(make_study) -> None:
    study = load_study(make_study())

    assert study.description == "Synthetic study"
    assert study.seeds == [1, 2, 3]
    assert study.axes == [
        Axis("activation", ["sigmoid", "tanh"], ["model.bias"]),
        Axis("dataset", ["a", "b"], ["model.bias"]),
    ]
    assert study.task == "classification"
    assert study.index.height == 12
    assert study.index.columns == ["dir", "variant", "seed", "activation", "dataset"]
    assert study.variants == ["sigmoid__a", "sigmoid__b", "tanh__a", "tanh__b"]


@pytest.mark.parametrize(
    ("report", "roles"),
    [
        ("", Roles(x="dataset", color=None, panels=["activation"])),
        ('[report]\nx = "activation"', Roles(x="activation", color=None, panels=["dataset"])),
        (
            '[report]\nx = "dataset"\ncolor = "activation"',
            Roles(x="dataset", color="activation", panels=[]),
        ),
        (
            '[report]\npanels = ["dataset"]',
            Roles(x=None, color=None, panels=["dataset", "activation"]),
        ),
    ],
)
def test_roles(make_study, report: str, roles: Roles) -> None:
    assert load_study(make_study(report=report)).roles == roles


def test_study_without_axes(make_study) -> None:
    study = load_study(make_study(axes={}))

    assert study.axes == []
    assert study.roles == Roles(x=None, color=None, panels=[])
    assert study.variants == ["base"]


def test_numeric_level_names_stay_strings(make_study) -> None:
    study = load_study(make_study(axes={"layers": ["0", "16", "16-16"]}))

    assert study.axes == [Axis("layers", ["0", "16", "16-16"], ["model.bias"])]
    assert study.index["dir"][0] == "runs/0/seed-1"


def test_reports_missing_files(tmp_path, make_study) -> None:
    with pytest.raises(StudyError, match="study.toml"):
        load_study(tmp_path)

    study_dir = make_study()
    (study_dir / "runs" / "index.csv").unlink()
    with pytest.raises(StudyError, match="compute generate"):
        load_study(study_dir)


def test_labels_replace_names(make_study) -> None:
    report = '[report]\nlabels = { dataset = "Zbiór", a = "A (100)" }'
    study = load_study(make_study(report=report))

    assert study.label("dataset") == "Zbiór"
    assert study.label("a") == "A (100)"
    assert study.label("b") == "b"
    assert study.label(None) is None


def test_split_defaults_to_test(make_study) -> None:
    assert load_study(make_study()).split == "test"


def test_reads_split(make_study) -> None:
    study = load_study(make_study(validation=True, report='[report]\nsplit = "validation"'))

    assert study.split == "validation"


def test_rejects_unknown_split(make_study) -> None:
    with pytest.raises(StudyError, match="report.split"):
        load_study(make_study(report='[report]\nsplit = "holdout"'))
