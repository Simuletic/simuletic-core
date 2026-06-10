from pathlib import Path

import tomllib


def test_pyproject_uses_correct_distribution_package_and_cli_names() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "simuletic-core"
    assert "rfdetr[train,loggers]>=1.2" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["scripts"] == {
        "simuletic-core": "simuletic_core.cli:app"
    }
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/simuletic_core"
    ]


def test_only_simuletic_core_source_package_exists() -> None:
    assert Path("src/simuletic_core").is_dir()
    assert not Path("src/simuletic").exists()
