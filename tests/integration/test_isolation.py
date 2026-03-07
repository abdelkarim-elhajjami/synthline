import pytest
from fastapi.testclient import TestClient
from api import app
import shutil
from pathlib import Path

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Ensure clean slate for session data."""
    sessions_dir = Path.cwd() / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    yield
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

def test_fm_isolation():
    """Verify that different sessions maintain separate FM states."""
    
    # 1. Prepare two distinct FMs
    fm_a = b"""<?xml version="1.0" encoding="UTF-8"?><extendedFeatureModel><struct><and name="SessionA"/></struct></extendedFeatureModel>"""
    fm_b = b"""<?xml version="1.0" encoding="UTF-8"?><extendedFeatureModel><struct><and name="SessionB"/></struct></extendedFeatureModel>"""
    
    headers_a = {"X-Session-ID": "test-session-a", "x-filename": "fm.xml"}
    headers_b = {"X-Session-ID": "test-session-b", "x-filename": "fm.xml"}
    
    # 2. Upload FM A to Session A
    resp_a = client.post(
        "/api/features/upload", 
        content=fm_a, 
        headers={"Content-Type": "application/xml", **headers_a}
    )
    assert resp_a.status_code == 200
    assert resp_a.json()["artefact_type"] == "SessionA"

    # 3. Upload FM B to Session B
    resp_b = client.post(
        "/api/features/upload", 
        content=fm_b, 
        headers={"Content-Type": "application/xml", **headers_b}
    )
    assert resp_b.status_code == 200
    assert resp_b.json()["artefact_type"] == "SessionB"
    
    # 4. Verify Session A still sees FM A, not B
    get_a = client.get("/api/features", headers=headers_a)
    assert get_a.status_code == 200
    assert get_a.json()["root"]["name"] == "SessionA"
    
    # 5. Verify Session B sees FM B
    get_b = client.get("/api/features", headers=headers_b)
    assert get_b.status_code == 200
    assert get_b.json()["root"]["name"] == "SessionB"
