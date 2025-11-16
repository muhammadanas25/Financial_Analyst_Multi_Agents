"""
Configuration module for Financial Analyst Multi-Agent System
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="fab-financial-analyst", env="LANGCHAIN_PROJECT")

    # Model Configuration
    primary_llm_model: str = Field(default="gpt-4-turbo-preview", env="PRIMARY_LLM_MODEL")
    fallback_llm_model: str = Field(default="gpt-3.5-turbo", env="FALLBACK_LLM_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )

    # Vector Database
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")  # chroma, weaviate, pinecone
    vector_db_path: str = Field(default="./chroma_db", env="VECTOR_DB_PATH")

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="fab_financial_db", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")

    # Document Processing
    chunk_size: int = Field(default=2048, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Retrieval Configuration
    hybrid_search_alpha: float = Field(default=0.3, env="HYBRID_SEARCH_ALPHA")
    retrieval_top_k: int = Field(default=10, env="RETRIEVAL_TOP_K")

    # Validation Configuration
    confidence_threshold: float = Field(default=0.70, env="CONFIDENCE_THRESHOLD")
    numerical_accuracy_threshold: float = Field(default=0.98, env="NUMERICAL_ACCURACY_THRESHOLD")
    faithfulness_threshold: float = Field(default=0.95, env="FAITHFULNESS_THRESHOLD")

    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Paths
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    logs_dir: Path = Path("./logs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    def ensure_directories(self):
        """Ensure required directories exist"""
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
