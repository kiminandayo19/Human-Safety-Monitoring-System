"""Smoke tests for health and root endpoints."""


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["docs"] == "/docs"


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_ready(client):
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
