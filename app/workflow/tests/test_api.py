from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_extract_rejects_unsupported_file():
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"not a real document", "text/plain")},
    )
    assert response.status_code == 400
