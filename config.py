import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama3-70b-8192")
    
    # Available Groq models
    AVAILABLE_MODELS: list = [
        "llama3-70b-8192",
        "llama3-8b-8192", 
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    
    # Evaluation settings
    DEFAULT_METRICS: list = ["accuracy", "faithfulness", "relevance", "toxicity"]
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_DIR: str = "./.cache"
    
    # LangGraph settings
    MAX_CONCURRENT: int = 5
    TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()