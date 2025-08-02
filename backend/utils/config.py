"""
Configuration Management for AstraFind
Uses Pydantic Settings for type-safe configuration with environment variable support
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic.networks import PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "AstraFind"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Security Settings
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000"], env="ALLOWED_ORIGINS")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database Settings
    DATABASE_URL: Optional[PostgresDsn] = Field(default=None, env="DATABASE_URL")
    DATABASE_HOST: str = Field(default="localhost", env="DATABASE_HOST")
    DATABASE_PORT: int = Field(default=5432, env="DATABASE_PORT")
    DATABASE_USER: str = Field(default="astrafind", env="DATABASE_USER")
    DATABASE_PASSWORD: str = Field(default="astrafind_password", env="DATABASE_PASSWORD")
    DATABASE_NAME: str = Field(default="astrafind", env="DATABASE_NAME")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Settings
    REDIS_URL: Optional[RedisDsn] = Field(default=None, env="REDIS_URL")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_POOL_SIZE: int = Field(default=20, env="REDIS_POOL_SIZE")
    
    # Elasticsearch Settings
    ELASTICSEARCH_URL: str = Field(default="http://localhost:9200", env="ELASTICSEARCH_URL")
    ELASTICSEARCH_INDEX_PREFIX: str = Field(default="astrafind", env="ELASTICSEARCH_INDEX_PREFIX")
    ELASTICSEARCH_TIMEOUT: int = Field(default=30, env="ELASTICSEARCH_TIMEOUT")
    ELASTICSEARCH_MAX_RETRIES: int = Field(default=3, env="ELASTICSEARCH_MAX_RETRIES")
    
    # RabbitMQ Settings
    RABBITMQ_URL: str = Field(
        default="amqp://astrafind:astrafind_password@localhost:5672//",
        env="RABBITMQ_URL"
    )
    
    # ML Model Settings
    ML_MODELS_PATH: str = Field(default="./ml-models", env="ML_MODELS_PATH")
    BERT_MODEL_NAME: str = Field(default="bert-base-multilingual-cased", env="BERT_MODEL_NAME")
    T5_MODEL_NAME: str = Field(default="t5-base", env="T5_MODEL_NAME")
    SENTENCE_TRANSFORMER_MODEL: str = Field(
        default="all-MiniLM-L6-v2", 
        env="SENTENCE_TRANSFORMER_MODEL"
    )
    
    # Crawler Settings
    CRAWLER_USER_AGENT: str = Field(
        default="AstraFind-Bot/1.0 (+https://astrafind.com/bot)", 
        env="CRAWLER_USER_AGENT"
    )
    CRAWLER_CONCURRENT_REQUESTS: int = Field(default=32, env="CRAWLER_CONCURRENT_REQUESTS")
    CRAWLER_DOWNLOAD_DELAY: float = Field(default=1.0, env="CRAWLER_DOWNLOAD_DELAY")
    CRAWLER_RANDOMIZE_DELAY: float = Field(default=0.5, env="CRAWLER_RANDOMIZE_DELAY")
    CRAWLER_RESPECT_ROBOTS_TXT: bool = Field(default=True, env="CRAWLER_RESPECT_ROBOTS_TXT")
    
    # Rate Limiting Settings
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    # Search Settings
    DEFAULT_SEARCH_RESULTS: int = Field(default=10, env="DEFAULT_SEARCH_RESULTS")
    MAX_SEARCH_RESULTS: int = Field(default=100, env="MAX_SEARCH_RESULTS")
    SEARCH_TIMEOUT_SECONDS: int = Field(default=5, env="SEARCH_TIMEOUT_SECONDS")
    
    # File Upload Settings
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/gif", "image/webp"],
        env="ALLOWED_IMAGE_TYPES"
    )
    
    # Cloud Storage Settings
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_S3_BUCKET: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    
    # GCP Settings
    GCP_PROJECT_ID: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    GCP_CREDENTIALS_PATH: Optional[str] = Field(default=None, env="GCP_CREDENTIALS_PATH")
    GCP_STORAGE_BUCKET: Optional[str] = Field(default=None, env="GCP_STORAGE_BUCKET")
    
    # Monitoring Settings
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Privacy & Compliance Settings
    GDPR_ENABLED: bool = Field(default=True, env="GDPR_ENABLED")
    CCPA_ENABLED: bool = Field(default=True, env="CCPA_ENABLED")
    DATA_RETENTION_DAYS: int = Field(default=365, env="DATA_RETENTION_DAYS")
    ANONYMIZE_IPS: bool = Field(default=True, env="ANONYMIZE_IPS")
    
    # AI Ethics Settings
    BIAS_DETECTION_ENABLED: bool = Field(default=True, env="BIAS_DETECTION_ENABLED")
    MISINFORMATION_FILTER_ENABLED: bool = Field(default=True, env="MISINFORMATION_FILTER_ENABLED")
    CONTENT_SAFETY_THRESHOLD: float = Field(default=0.8, env="CONTENT_SAFETY_THRESHOLD")
    
    @field_validator("DATABASE_URL", mode='before')
    @classmethod
    def build_database_url(cls, v, info):
        """Build database URL from individual components if not provided"""
        if v:
            return v
        
        data = info.data
        # Only build URL if we have the required components
        if data.get("DATABASE_HOST") and data.get("DATABASE_NAME"):
            return PostgresDsn.build(
                scheme="postgresql",
                username=data.get("DATABASE_USER"),
                password=data.get("DATABASE_PASSWORD"),
                host=data.get("DATABASE_HOST"),
                port=data.get("DATABASE_PORT"),
                path=f"/{data.get('DATABASE_NAME')}"
            )
        return None
    
    @field_validator("REDIS_URL", mode='before')
    @classmethod
    def build_redis_url(cls, v, info):
        """Build Redis URL from individual components if not provided"""
        if v:
            return v
        
        data = info.data
        # Only build URL if we have the required components
        if data.get("REDIS_HOST"):
            password = data.get("REDIS_PASSWORD")
            auth = f":{password}@" if password else ""
            port = data.get("REDIS_PORT", 6379)
            db = data.get("REDIS_DB", 0)
            
            return f"redis://{auth}{data.get('REDIS_HOST')}:{port}/{db}"
        return None
    
    @field_validator("ALLOWED_HOSTS", "ALLOWED_ORIGINS", "ALLOWED_IMAGE_TYPES", mode='before')
    @classmethod
    def parse_list_from_string(cls, v):
        """Parse comma-separated string into list"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


# Create settings instance
settings = Settings()


def get_database_url() -> str:
    """Get database URL for SQLAlchemy"""
    return str(settings.DATABASE_URL)


def get_redis_url() -> str:
    """Get Redis URL for connection"""
    return str(settings.REDIS_URL)


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.ENVIRONMENT.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return settings.ENVIRONMENT.lower() == "development"


def is_testing() -> bool:
    """Check if running in testing environment"""
    return settings.ENVIRONMENT.lower() == "testing"