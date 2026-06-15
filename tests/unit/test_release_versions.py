import ast
import json
import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _sdk_version() -> str:
    module = ast.parse((ROOT / "synthline" / "__init__.py").read_text())
    return next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )


def test_release_versions_are_synchronized():
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text()
    )["project"]["version"]
    web_version = json.loads(
        (ROOT / "web" / "package.json").read_text()
    )["version"]

    assert project_version == _sdk_version() == web_version
