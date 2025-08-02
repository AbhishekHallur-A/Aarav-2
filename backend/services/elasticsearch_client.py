import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import (
    ConnectionError, TransportError, NotFoundError,
    RequestError, ConflictError
)
import structlog
from ..utils.config import settings

logger = structlog.get_logger(__name__)

class ElasticsearchManager:
    """
    Elasticsearch client manager for AstraFind search engine
    """
    
    _instance: Optional['ElasticsearchManager'] = None
    _client: Optional[AsyncElasticsearch] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def initialize(cls):
        """Initialize Elasticsearch client"""
        if cls._client is None:
            try:
                cls._client = AsyncElasticsearch(
                    hosts=[settings.ELASTICSEARCH_URL],
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True,
                    verify_certs=False,  # Set to True in production with proper certs
                )
                
                # Test connection
                await cls._client.ping()
                logger.info("Elasticsearch connection established")
                
                # Create default indices
                await cls.create_default_indices()
                
            except Exception as e:
                logger.error(f"Failed to initialize Elasticsearch: {e}")
                raise
    
    @classmethod
    async def close(cls):
        """Close Elasticsearch client"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            logger.info("Elasticsearch connection closed")
    
    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        """Get Elasticsearch client instance"""
        if cls._client is None:
            await cls.initialize()
        return cls._client
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check Elasticsearch health"""
        try:
            client = await cls.get_client()
            response = await client.cluster.health()
            return response['status'] in ['green', 'yellow']
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    @classmethod
    async def create_default_indices(cls):
        """Create default indices with mappings"""
        client = await cls.get_client()
        
        # Documents index
        documents_mapping = {
            "mappings": {
                "properties": {
                    "url": {"type": "keyword"},
                    "url_hash": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {
                                "type": "completion",
                                "analyzer": "simple"
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "meta_description": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "language": {"type": "keyword"},
                    "content_type": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text"},
                            "label": {"type": "keyword"},
                            "confidence": {"type": "float"}
                        }
                    },
                    "topics": {"type": "keyword"},
                    "word_count": {"type": "integer"},
                    "content_quality_score": {"type": "float"},
                    "readability_score": {"type": "float"},
                    "sentiment_score": {"type": "float"},
                    "misinformation_score": {"type": "float"},
                    "bias_score": {"type": "float"},
                    "page_rank_score": {"type": "float"},
                    "popularity_score": {"type": "float"},
                    "freshness_score": {"type": "float"},
                    "source_credibility": {"type": "float"},
                    "crawled_at": {"type": "date"},
                    "last_modified": {"type": "date"},
                    "is_indexed": {"type": "boolean"},
                    "is_blocked": {"type": "boolean"},
                    "search_vector": {"type": "text"}
                }
            },
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "content_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "stemmer",
                                "synonym"
                            ]
                        },
                        "search_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        },
                        "synonym": {
                            "type": "synonym",
                            "synonyms": [
                                "ai,artificial intelligence",
                                "ml,machine learning",
                                "nlp,natural language processing"
                            ]
                        }
                    }
                }
            }
        }
        
        # Search queries index
        queries_mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "session_id": {"type": "keyword"},
                    "query_text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "query_type": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "results_count": {"type": "integer"},
                    "response_time_ms": {"type": "integer"},
                    "clicked_results": {"type": "object"},
                    "result_satisfaction": {"type": "float"},
                    "user_location": {"type": "keyword"},
                    "user_device": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "anonymized": {"type": "boolean"}
                }
            },
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        }
        
        # Create indices
        await cls._create_index_if_not_exists("documents", documents_mapping)
        await cls._create_index_if_not_exists("search_queries", queries_mapping)
        
        logger.info("Default Elasticsearch indices created")
    
    @classmethod
    async def _create_index_if_not_exists(cls, index_name: str, mapping: Dict):
        """Create index if it doesn't exist"""
        client = await cls.get_client()
        
        try:
            exists = await client.indices.exists(index=index_name)
            if not exists:
                await client.indices.create(index=index_name, body=mapping)
                logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                logger.info(f"Elasticsearch index already exists: {index_name}")
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            raise
    
    @classmethod
    async def index_document(cls, document: Dict[str, Any]) -> bool:
        """Index a document"""
        client = await cls.get_client()
        
        try:
            # Generate document ID from URL hash
            doc_id = document.get('url_hash') or hashlib.sha256(
                document['url'].encode()
            ).hexdigest()
            
            # Prepare document for indexing
            es_doc = cls._prepare_document_for_indexing(document)
            
            # Index document
            response = await client.index(
                index="documents",
                id=doc_id,
                body=es_doc,
                refresh=True
            )
            
            logger.info(f"Indexed document: {document.get('url', 'unknown')}")
            return response['result'] in ['created', 'updated']
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False
    
    @classmethod
    def _prepare_document_for_indexing(cls, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document for Elasticsearch indexing"""
        # Remove None values and prepare data types
        es_doc = {}
        
        for key, value in document.items():
            if value is not None:
                if key in ['crawled_at', 'last_modified'] and isinstance(value, datetime):
                    es_doc[key] = value.isoformat()
                elif key == 'entities' and isinstance(value, dict):
                    # Convert entities to nested format
                    es_doc[key] = [
                        {
                            'text': ent_text,
                            'label': ent_data.get('label', ''),
                            'confidence': ent_data.get('confidence', 0.0)
                        }
                        for ent_text, ent_data in value.items()
                    ]
                else:
                    es_doc[key] = value
        
        return es_doc
    
    @classmethod
    async def search(
        cls,
        query: str,
        size: int = 10,
        from_: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        language: str = 'en',
        include_blocked: bool = False
    ) -> Dict[str, Any]:
        """Search documents"""
        client = await cls.get_client()
        
        try:
            # Build search query
            search_body = cls._build_search_query(
                query, filters, sort, language, include_blocked
            )
            
            # Execute search
            response = await client.search(
                index="documents",
                body=search_body,
                size=size,
                from_=from_
            )
            
            # Process results
            return cls._process_search_results(response)
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"hits": [], "total": 0, "took": 0}
    
    @classmethod
    def _build_search_query(
        cls,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        language: str = 'en',
        include_blocked: bool = False
    ) -> Dict[str, Any]:
        """Build Elasticsearch search query"""
        
        # Base query structure
        search_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "must_not": [],
                    "should": []
                }
            }
        }
        
        # Main text search
        if query.strip():
            search_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "meta_description^2",
                        "content",
                        "keywords^2"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            })
            
            # Boost exact phrase matches
            search_body["query"]["bool"]["should"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["title^5", "meta_description^3"],
                    "type": "phrase",
                    "boost": 2
                }
            })
        else:
            # Match all if no query
            search_body["query"]["bool"]["must"].append({
                "match_all": {}
            })
        
        # Language filter
        if language:
            search_body["query"]["bool"]["filter"].append({
                "term": {"language": language}
            })
        
        # Block filter
        if not include_blocked:
            search_body["query"]["bool"]["filter"].append({
                "term": {"is_blocked": False}
            })
        
        # Only indexed documents
        search_body["query"]["bool"]["filter"].append({
            "term": {"is_indexed": True}
        })
        
        # Apply additional filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    search_body["query"]["bool"]["filter"].append({
                        "terms": {field: value}
                    })
                elif isinstance(value, dict) and "range" in value:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {field: value["range"]}
                    })
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {field: value}
                    })
        
        # Default sorting
        if not sort:
            sort = [
                {"_score": {"order": "desc"}},
                {"content_quality_score": {"order": "desc"}},
                {"freshness_score": {"order": "desc"}},
                {"crawled_at": {"order": "desc"}}
            ]
        
        search_body["sort"] = sort
        
        # Add highlighting
        search_body["highlight"] = {
            "fields": {
                "title": {"fragment_size": 150, "number_of_fragments": 1},
                "content": {"fragment_size": 200, "number_of_fragments": 3},
                "meta_description": {"fragment_size": 150, "number_of_fragments": 1}
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"]
        }
        
        return search_body
    
    @classmethod
    def _process_search_results(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process Elasticsearch search results"""
        hits = response.get('hits', {})
        
        processed_results = {
            "hits": [],
            "total": hits.get('total', {}).get('value', 0),
            "took": response.get('took', 0),
            "max_score": hits.get('max_score', 0)
        }
        
        for hit in hits.get('hits', []):
            source = hit['_source']
            
            result = {
                "id": hit['_id'],
                "score": hit['_score'],
                "url": source.get('url'),
                "title": source.get('title'),
                "meta_description": source.get('meta_description'),
                "content_snippet": cls._get_content_snippet(source.get('content', ''), 200),
                "domain": source.get('domain'),
                "language": source.get('language'),
                "content_type": source.get('content_type'),
                "keywords": source.get('keywords', []),
                "topics": source.get('topics', []),
                "content_quality_score": source.get('content_quality_score', 0),
                "crawled_at": source.get('crawled_at'),
                "highlights": hit.get('highlight', {})
            }
            
            processed_results["hits"].append(result)
        
        return processed_results
    
    @classmethod
    def _get_content_snippet(cls, content: str, max_length: int = 200) -> str:
        """Get content snippet for search results"""
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        # Try to break at word boundary
        snippet = content[:max_length]
        last_space = snippet.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can break near the end
            snippet = snippet[:last_space]
        
        return snippet + "..."
    
    @classmethod
    async def get_suggestions(cls, query: str, size: int = 10) -> List[str]:
        """Get search suggestions"""
        client = await cls.get_client()
        
        try:
            search_body = {
                "suggest": {
                    "title_suggest": {
                        "prefix": query,
                        "completion": {
                            "field": "title.suggest",
                            "size": size,
                            "skip_duplicates": True
                        }
                    }
                }
            }
            
            response = await client.search(
                index="documents",
                body=search_body
            )
            
            suggestions = []
            for option in response.get('suggest', {}).get('title_suggest', [{}])[0].get('options', []):
                suggestions.append(option['text'])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
    
    @classmethod
    async def get_trending_topics(cls, days: int = 7, size: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics from search queries"""
        client = await cls.get_client()
        
        try:
            # Get date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            search_body = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "created_at": {
                                        "gte": start_date.isoformat(),
                                        "lte": end_date.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "aggs": {
                    "trending_queries": {
                        "terms": {
                            "field": "query_text.keyword",
                            "size": size,
                            "order": {"_count": "desc"}
                        }
                    }
                }
            }
            
            response = await client.search(
                index="search_queries",
                body=search_body
            )
            
            trending = []
            for bucket in response.get('aggregations', {}).get('trending_queries', {}).get('buckets', []):
                trending.append({
                    "query": bucket['key'],
                    "count": bucket['doc_count']
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    @classmethod
    async def record_search_query(cls, query_data: Dict[str, Any]) -> bool:
        """Record a search query for analytics"""
        client = await cls.get_client()
        
        try:
            # Add timestamp if not present
            if 'created_at' not in query_data:
                query_data['created_at'] = datetime.utcnow().isoformat()
            
            response = await client.index(
                index="search_queries",
                body=query_data,
                refresh=False  # Don't refresh immediately for performance
            )
            
            return response['result'] in ['created', 'updated']
            
        except Exception as e:
            logger.error(f"Error recording search query: {e}")
            return False
    
    @classmethod
    async def update_document_scores(cls, document_id: str, scores: Dict[str, float]) -> bool:
        """Update document scores (quality, popularity, etc.)"""
        client = await cls.get_client()
        
        try:
            response = await client.update(
                index="documents",
                id=document_id,
                body={
                    "doc": scores
                },
                refresh=False
            )
            
            return response['result'] == 'updated'
            
        except NotFoundError:
            logger.warning(f"Document not found for score update: {document_id}")
            return False
        except Exception as e:
            logger.error(f"Error updating document scores: {e}")
            return False
    
    @classmethod
    async def delete_document(cls, document_id: str) -> bool:
        """Delete a document from the index"""
        client = await cls.get_client()
        
        try:
            response = await client.delete(
                index="documents",
                id=document_id,
                refresh=True
            )
            
            return response['result'] == 'deleted'
            
        except NotFoundError:
            logger.warning(f"Document not found for deletion: {document_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    @classmethod
    async def bulk_index_documents(cls, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple documents"""
        client = await cls.get_client()
        
        try:
            actions = []
            for doc in documents:
                doc_id = doc.get('url_hash') or hashlib.sha256(
                    doc['url'].encode()
                ).hexdigest()
                
                action = {
                    "_index": "documents",
                    "_id": doc_id,
                    "_source": cls._prepare_document_for_indexing(doc)
                }
                actions.append(action)
            
            if not actions:
                return {"success": 0, "error": 0}
            
            # Execute bulk operation
            response = await client.bulk(body=actions, refresh=False)
            
            # Count successes and errors
            success_count = 0
            error_count = 0
            
            for item in response['items']:
                if 'index' in item:
                    if item['index'].get('status') in [200, 201]:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Bulk index error: {item['index'].get('error')}")
            
            logger.info(f"Bulk indexed {success_count} documents, {error_count} errors")
            return {"success": success_count, "error": error_count}
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")
            return {"success": 0, "error": len(documents)}
    
    @classmethod
    async def get_index_stats(cls) -> Dict[str, Any]:
        """Get index statistics"""
        client = await cls.get_client()
        
        try:
            response = await client.indices.stats(index="documents,search_queries")
            
            stats = {}
            for index_name, index_stats in response.get('indices', {}).items():
                stats[index_name] = {
                    "doc_count": index_stats['total']['docs']['count'],
                    "store_size": index_stats['total']['store']['size_in_bytes'],
                    "primary_shards": index_stats['primaries']['docs']['count']
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    @classmethod
    async def cleanup_old_queries(cls, days: int = 90) -> int:
        """Clean up old search queries for compliance"""
        client = await cls.get_client()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            response = await client.delete_by_query(
                index="search_queries",
                body={
                    "query": {
                        "range": {
                            "created_at": {
                                "lt": cutoff_date.isoformat()
                            }
                        }
                    }
                }
            )
            
            deleted_count = response.get('deleted', 0)
            logger.info(f"Cleaned up {deleted_count} old search queries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old queries: {e}")
            return 0