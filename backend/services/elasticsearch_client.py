"""
Elasticsearch Client Service
Handles Elasticsearch connections and operations
"""

import logging
from typing import Optional, Dict, Any, List
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from utils.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchManager:
    """Elasticsearch connection manager"""
    
    _client: Optional[AsyncElasticsearch] = None
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize Elasticsearch client"""
        try:
            cls._client = AsyncElasticsearch(
                [settings.ELASTICSEARCH_URL],
                timeout=settings.ELASTICSEARCH_TIMEOUT,
                max_retries=settings.ELASTICSEARCH_MAX_RETRIES,
                retry_on_timeout=True
            )
            # Test connection
            await cls._client.ping()
            logger.info("Elasticsearch client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")
            raise
    
    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        """Get Elasticsearch client"""
        if cls._client is None:
            await cls.initialize()
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        """Close Elasticsearch client"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            logger.info("Elasticsearch client closed")
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check Elasticsearch health"""
        try:
            client = await cls.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    @classmethod
    async def create_default_indices(cls) -> None:
        """Create default indices if they don't exist"""
        try:
            client = await cls.get_client()
            
            # Create documents index
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            if not await client.indices.exists(index=index_name):
                await client.indices.create(
                    index=index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "url": {"type": "keyword"},
                                "title": {"type": "text", "analyzer": "standard"},
                                "content": {"type": "text", "analyzer": "standard"},
                                "language": {"type": "keyword"},
                                "content_type": {"type": "keyword"},
                                "crawl_time": {"type": "date"},
                                "embedding": {"type": "dense_vector", "dims": 768}
                            }
                        }
                    }
                )
                logger.info(f"Created index: {index_name}")
            
            # Create search logs index
            logs_index = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_search_logs"
            if not await client.indices.exists(index=logs_index):
                await client.indices.create(
                    index=logs_index,
                    body={
                        "mappings": {
                            "properties": {
                                "query": {"type": "text"},
                                "user_id": {"type": "keyword"},
                                "timestamp": {"type": "date"},
                                "results_count": {"type": "integer"},
                                "search_time_ms": {"type": "float"}
                            }
                        }
                    }
                )
                logger.info(f"Created index: {logs_index}")
                
        except Exception as e:
            logger.error(f"Failed to create default indices: {e}")
            raise