"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    Values are read from environment variables or a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "Human Safety Monitoring System"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    API_V1_PREFIX: str = "/api/v1"

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- CORS ---
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    # --- Computer Vision / Detection ---
    MODEL_PATH: str = Field(default="models/safety_detector.pt")
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    DEVICE: str = Field(default="cpu")  # "cpu" or "cuda"

    # --- Safety rules ---
    # Minimum confidence before a safety event is escalated to an alert.
    ALERT_CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Logging ---
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
