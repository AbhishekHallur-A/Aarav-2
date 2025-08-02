"""
Elasticsearch Manager for AstraFind
Handles Elasticsearch connections and operations
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import structlog

from ..utils.config import settings

logger = structlog.get_logger(__name__)

class ElasticsearchManager:
    """Manages Elasticsearch connections and operations"""
    
    _client: Optional[AsyncElasticsearch] = None
    _initialized: bool = False
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize Elasticsearch connection"""
        if cls._initialized:
            return
            
        try:
            cls._client = AsyncElasticsearch(
                [settings.ELASTICSEARCH_URL],
                timeout=settings.ELASTICSEARCH_TIMEOUT,
                max_retries=settings.ELASTICSEARCH_MAX_RETRIES,
                retry_on_timeout=True
            )
            
            # Test connection
            await cls._client.ping()
            cls._initialized = True
            logger.info("Elasticsearch connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise
    
    @classmethod
    async def close(cls) -> None:
        """Close Elasticsearch connection"""
        if cls._client:
            await cls._client.close()
        cls._initialized = False
        logger.info("Elasticsearch connection closed")
    
    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        """Get Elasticsearch client"""
        if not cls._initialized:
            await cls.initialize()
        return cls._client
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check Elasticsearch health"""
        try:
            if not cls._initialized:
                return False
            
            client = await cls.get_client()
            health = await client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    @classmethod
    async def create_default_indices(cls) -> None:
        """Create default Elasticsearch indices"""
        if not cls._initialized:
            await cls.initialize()
        
        client = await cls.get_client()
        
        # Documents index
        documents_mapping = {
            "mappings": {
                "properties": {
                    "url": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "language": {"type": "keyword"},
                    "content_type": {"type": "keyword"},
                    "quality_score": {"type": "float"},
                    "crawl_time": {"type": "date"},
                    "domain": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384  # Default for all-MiniLM-L6-v2
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "content_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            }
        }
        
        # Create documents index
        index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
        try:
            await client.indices.create(index=index_name, body=documents_mapping)
            logger.info(f"Created index: {index_name}")
        except Exception as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"Failed to create index {index_name}: {e}")
        
        # Search analytics index
        analytics_mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "user_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "results_count": {"type": "integer"},
                    "search_time_ms": {"type": "float"},
                    "clicked_results": {"type": "keyword"},
                    "session_id": {"type": "keyword"}
                }
            }
        }
        
        analytics_index = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_analytics"
        try:
            await client.indices.create(index=analytics_index, body=analytics_mapping)
            logger.info(f"Created index: {analytics_index}")
        except Exception as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"Failed to create index {analytics_index}: {e}")
    
    @classmethod
    async def index_document(cls, document: Dict[str, Any], doc_id: str) -> bool:
        """Index a document in Elasticsearch"""
        try:
            client = await cls.get_client()
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            
            response = await client.index(
                index=index_name,
                id=doc_id,
                body=document
            )
            
            return response['result'] in ['created', 'updated']
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return False
    
    @classmethod
    async def search_documents(
        cls,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0
    ) -> Dict[str, Any]:
        """Search documents in Elasticsearch"""
        try:
            client = await cls.get_client()
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2", "content", "description"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": size,
                "from": from_,
                "highlight": {
                    "fields": {
                        "content": {"fragment_size": 150, "number_of_fragments": 3},
                        "title": {"fragment_size": 100, "number_of_fragments": 1}
                    }
                }
            }
            
            # Add filters
            if filters:
                search_body["query"]["bool"]["filter"] = []
                for field, value in filters.items():
                    search_body["query"]["bool"]["filter"].append(
                        {"term": {field: value}}
                    )
            
            response = await client.search(index=index_name, body=search_body)
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"hits": {"total": {"value": 0}, "hits": []}}
    
    @classmethod
    async def delete_document(cls, doc_id: str) -> bool:
        """Delete a document from Elasticsearch"""
        try:
            client = await cls.get_client()
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            
            response = await client.delete(index=index_name, id=doc_id)
            return response['result'] == 'deleted'
        except NotFoundError:
            return True  # Document doesn't exist
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    @classmethod
    async def get_document(cls, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document from Elasticsearch"""
        try:
            client = await cls.get_client()
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            
            response = await client.get(index=index_name, id=doc_id)
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    @classmethod
    async def update_document(cls, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document in Elasticsearch"""
        try:
            client = await cls.get_client()
            index_name = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_documents"
            
            response = await client.update(
                index=index_name,
                id=doc_id,
                body={"doc": updates}
            )
            
            return response['result'] == 'updated'
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False