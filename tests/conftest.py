"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient that triggers app lifespan (model load)."""
    app = create_app()
    with TestClient(app) as c:
        yield c
