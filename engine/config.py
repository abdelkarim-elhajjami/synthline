from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings managed by Pydantic."""
    
    # API Keys
    DEEPSEEK_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # LLM Configuration
    OLLAMA_BASE_URL: Optional[str] = None
    
    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://web:3000"]
    
    # Logging
    LOG_LEVEL: str = "info"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
