from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from api import app
from dependencies import get_dependencies

client = TestClient(app)


def _mock_deps():
    deps = MagicMock()
    deps.features = {
        "root": {"name": "Artefact"},
        "artefact_type": "Requirement",
        "source_path": "config/fm.xml",
    }
    deps.update_feature_model.return_value = deps.features
    deps.update_glossary.return_value = {"entries": 2, "replaced": False}
    return deps


def test_get_features_success():
    mock_deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    try:
        response = client.get("/api/features")
        assert response.status_code == 200
        assert response.json()["artefact_type"] == "Requirement"
    finally:
        app.dependency_overrides = {}


def test_upload_features_success():
    mock_deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    try:
        xml_payload = b"""<?xml version="1.0" encoding="UTF-8"?><extendedFeatureModel><struct><and name="Artefact"/></struct></extendedFeatureModel>"""
        response = client.post(
            "/api/features/upload",
            content=xml_payload,
            headers={"Content-Type": "application/xml", "x-filename": "fm.xml"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert data["artefact_type"] == "Requirement"
        mock_deps.update_feature_model.assert_called_once()
    finally:
        app.dependency_overrides = {}


def test_upload_features_rejects_non_xml():
    mock_deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    try:
        response = client.post(
            "/api/features/upload",
            content=b"not xml",
            headers={"Content-Type": "text/plain", "x-filename": "fm.txt"},
        )
        assert response.status_code == 400
        assert "Only .xml files are supported" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}


def test_upload_glossary_success():
    mock_deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    try:
        yaml_payload = b"TermA: Definition A\nTermB: Definition B\n"
        response = client.post(
            "/api/glossary/upload",
            content=yaml_payload,
            headers={"Content-Type": "text/yaml", "x-filename": "glossary.yaml"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert data["entries"] == 2
        assert data["replaced"] is False
        mock_deps.update_glossary.assert_called_once()
    finally:
        app.dependency_overrides = {}


def test_upload_glossary_rejects_invalid_extension():
    mock_deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    try:
        response = client.post(
            "/api/glossary/upload",
            content=b"Term: Definition",
            headers={"Content-Type": "text/plain", "x-filename": "glossary.txt"},
        )
        assert response.status_code == 400
        assert "Only .yaml and .yml files are supported" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}
