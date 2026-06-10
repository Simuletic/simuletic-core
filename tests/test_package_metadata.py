from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def test_pyproject_uses_correct_distribution_package_and_cli_names() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "simuletic-vision"
    assert "rfdetr>=1.2" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["scripts"] == {
        "simuletic-vision": "simuletic_vision.cli:app"
    }
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/simuletic_vision"
    ]


def test_only_simuletic_vision_source_package_exists() -> None:
    assert Path("src/simuletic_vision").is_dir()
    assert not Path("src/simuletic").exists()
