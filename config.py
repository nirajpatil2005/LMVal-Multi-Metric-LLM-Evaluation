import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Default models
    DEFAULT_GROQ_MODEL: str = os.getenv("DEFAULT_GROQ_MODEL", "openai/gpt-oss-20b")
    DEFAULT_OPENAI_MODEL: str = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o")
    
    # Available models
    AVAILABLE_GROQ_MODELS: list = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-guard-4-12b"
    ]
    
    AVAILABLE_OPENAI_MODELS: list = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    # Evaluation settings
    DEFAULT_METRICS: list = ["accuracy", "faithfulness", "relevance", "toxicity"]
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_DIR: str = "./.cache"
    
    # LangGraph settings
    MAX_CONCURRENT: int = 5
    TIMEOUT: int = 30
    
    # API Provider
    DEFAULT_API_PROVIDER: str = os.getenv("DEFAULT_API_PROVIDER", "groq")
    
    class Config:
        env_file = ".env"

settings = Settings()