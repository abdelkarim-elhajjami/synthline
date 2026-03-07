from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file="../.env",
        case_sensitive=True,
        extra="ignore",
    )
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    # LLM Configuration
    OLLAMA_BASE_URL: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://web:3000"]
    # Logging
    LOG_LEVEL: str = "info"
    # Feature model configuration
    FM_XML_PATH: Optional[str] = "config/uploaded_fm.xml"
    GLOSSARY_PATH: Optional[str] = None

settings = Settings()
