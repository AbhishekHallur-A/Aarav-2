"""
Search API Routes
Handles text, voice, and image search requests
"""

import asyncio
from typing import List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
import structlog

from ...services.search.hybrid_ranker import HybridSearchRanker, SearchQuery, SearchResult
from ...services.indexer.semantic_indexer import SemanticIndexer
from ...services.elasticsearch_client import ElasticsearchManager
from ...services.ml.image_processor import ImageProcessor
from ...services.ml.voice_processor import VoiceProcessor
from ...utils.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

# Pydantic models for request/response
class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    language: Optional[str] = Field(default="en", description="Query language code")
    content_type_filter: Optional[str] = Field(default=None, description="Filter by content type")
    time_filter: Optional[str] = Field(default=None, description="Filter by time period")
    safe_search: bool = Field(default=True, description="Enable safe search filtering")
    personalized: bool = Field(default=True, description="Enable personalized results")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    include_similar: bool = Field(default=False, description="Include similar documents")

class SearchResultResponse(BaseModel):
    document_id: str
    url: str
    title: str
    description: str
    content_snippet: str
    content_type: str
    language: str
    quality_score: float
    final_score: float
    crawl_time: Optional[datetime]
    similar_documents: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    total_results: int
    search_time_ms: float
    results: List[SearchResultResponse]
    suggestions: Optional[List[str]] = None
    trending_topics: Optional[List[str]] = None

class VoiceSearchRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(default="wav", description="Audio format")
    language: Optional[str] = Field(default="en-US", description="Speech recognition language")

class ImageSearchRequest(BaseModel):
    search_type: str = Field(default="similarity", description="Type of image search")
    include_objects: bool = Field(default=True, description="Include object detection results")
    include_text: bool = Field(default=True, description="Include OCR text extraction")

# Dependency injection
async def get_hybrid_ranker() -> HybridSearchRanker:
    """Get hybrid search ranker instance"""
    # In production, this would be a singleton managed by dependency injection
    ranker = HybridSearchRanker()
    if not ranker.initialized:
        await ranker.initialize()
    return ranker

async def get_semantic_indexer() -> SemanticIndexer:
    """Get semantic indexer instance"""
    indexer = SemanticIndexer()
    if not indexer.initialized:
        await indexer.initialize()
    return indexer

async def get_image_processor() -> ImageProcessor:
    """Get image processor instance"""
    processor = ImageProcessor()
    if not processor.initialized:
        await processor.initialize()
    return processor

async def get_voice_processor() -> VoiceProcessor:
    """Get voice processor instance"""
    processor = VoiceProcessor()
    if not processor.initialized:
        await processor.initialize()
    return processor

@router.post("/text", response_model=SearchResponse)
async def search_text(
    request: TextSearchRequest,
    user_id: Optional[str] = Query(default=None, description="User ID for personalization"),
    ranker: HybridSearchRanker = Depends(get_hybrid_ranker),
    indexer: SemanticIndexer = Depends(get_semantic_indexer)
):
    """
    Perform text-based search with hybrid ranking
    """
    start_time = datetime.utcnow()
    
    try:
        # Create search query
        search_query = SearchQuery(
            text=request.query,
            user_id=user_id,
            language=request.language,
            content_type_filter=request.content_type_filter,
            time_filter=request.time_filter,
            safe_search=request.safe_search,
            personalized=request.personalized
        )
        
        # Get candidate documents from Elasticsearch
        es_results = await ElasticsearchManager.search(
            query=request.query,
            filters={
                'language': request.language,
                'content_type': request.content_type_filter
            },
            size=request.max_results * 2  # Get more candidates for re-ranking
        )
        
        if not es_results:
            return SearchResponse(
                query=request.query,
                total_results=0,
                search_time_ms=0.0,
                results=[],
                suggestions=[],
                trending_topics=[]
            )
        
        # Re-rank using hybrid approach
        ranked_results = await ranker.rank_results(
            query=search_query,
            candidate_documents=es_results,
            max_results=request.max_results
        )
        
        # Convert to response format
        response_results = []
        for result in ranked_results:
            # Get content snippet
            content_snippet = result.content[:300] + "..." if len(result.content) > 300 else result.content
            
            # Get similar documents if requested
            similar_docs = None
            if request.include_similar:
                similar_results = await indexer.find_similar_documents(
                    document_id=result.document_id,
                    top_k=3
                )
                similar_docs = [doc.url for doc, _ in similar_results]
            
            response_results.append(SearchResultResponse(
                document_id=result.document_id,
                url=result.url,
                title=result.title,
                description=result.description,
                content_snippet=content_snippet,
                content_type=result.content_type,
                language=result.language,
                quality_score=result.quality_score,
                final_score=result.final_score,
                crawl_time=result.crawl_time,
                similar_documents=similar_docs
            ))
        
        # Get search suggestions and trending topics
        suggestions = await get_search_suggestions(request.query)
        trending_topics = await get_trending_topics()
        
        # Calculate search time
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(f"Text search completed", 
                   query=request.query, 
                   results_count=len(response_results),
                   search_time_ms=search_time)
        
        return SearchResponse(
            query=request.query,
            total_results=len(response_results),
            search_time_ms=search_time,
            results=response_results,
            suggestions=suggestions,
            trending_topics=trending_topics
        )
        
    except Exception as e:
        logger.error(f"Error in text search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search service error")

@router.post("/voice", response_model=SearchResponse)
async def search_voice(
    request: VoiceSearchRequest,
    user_id: Optional[str] = Query(default=None),
    voice_processor: VoiceProcessor = Depends(get_voice_processor),
    ranker: HybridSearchRanker = Depends(get_hybrid_ranker)
):
    """
    Perform voice-based search with speech-to-text conversion
    """
    try:
        # Convert voice to text
        transcript = await voice_processor.speech_to_text(
            audio_data=request.audio_data,
            format=request.format,
            language=request.language
        )
        
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Perform text search with transcript
        text_request = TextSearchRequest(
            query=transcript,
            language=request.language[:2] if request.language else "en"  # Convert en-US to en
        )
        
        return await search_text(text_request, user_id, ranker)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice search service error")

@router.post("/image", response_model=SearchResponse)
async def search_image(
    image: UploadFile = File(..., description="Image file for search"),
    search_type: str = Form(default="similarity", description="Type of image search"),
    include_objects: bool = Form(default=True),
    include_text: bool = Form(default=True),
    user_id: Optional[str] = Query(default=None),
    image_processor: ImageProcessor = Depends(get_image_processor),
    ranker: HybridSearchRanker = Depends(get_hybrid_ranker)
):
    """
    Perform image-based search with object detection and OCR
    """
    try:
        # Validate image file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        if image.size and image.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Image file too large")
        
        # Read image data
        image_data = await image.read()
        
        # Process image
        analysis_result = await image_processor.analyze_image(
            image_data=image_data,
            include_objects=include_objects,
            include_text=include_text
        )
        
        # Generate search query from image analysis
        query_parts = []
        
        # Add detected objects
        if analysis_result.objects:
            object_names = [obj['name'] for obj in analysis_result.objects[:5]]  # Top 5 objects
            query_parts.extend(object_names)
        
        # Add extracted text
        if analysis_result.text and include_text:
            query_parts.append(analysis_result.text[:100])  # Limit text length
        
        # Add tags
        if analysis_result.tags:
            query_parts.extend(analysis_result.tags[:3])  # Top 3 tags
        
        if not query_parts:
            raise HTTPException(status_code=400, detail="Could not extract searchable content from image")
        
        search_query = " ".join(query_parts)
        
        # Perform semantic search if we have image embeddings
        if hasattr(analysis_result, 'embedding') and analysis_result.embedding is not None:
            # Use semantic similarity search
            indexer = await get_semantic_indexer()
            semantic_results = await indexer.semantic_search(
                query=search_query,
                top_k=20
            )
            
            # Convert semantic results to search results format
            candidate_docs = []
            for doc, similarity in semantic_results:
                candidate_docs.append({
                    '_id': doc.document_id,
                    '_source': {
                        'url': doc.url,
                        'title': doc.title,
                        'content': {'text': doc.content},
                        'description': doc.content[:200],
                        'content_type': doc.content_type,
                        'language': doc.language,
                        'content_score': doc.quality_score,
                        'crawl_time': doc.processed_at.isoformat() if doc.processed_at else None
                    }
                })
        else:
            # Fall back to text search
            es_results = await ElasticsearchManager.search(
                query=search_query,
                size=20
            )
            candidate_docs = es_results
        
        # Re-rank results
        search_query_obj = SearchQuery(
            text=search_query,
            user_id=user_id
        )
        
        ranked_results = await ranker.rank_results(
            query=search_query_obj,
            candidate_documents=candidate_docs,
            max_results=10
        )
        
        # Convert to response format
        response_results = []
        for result in ranked_results:
            content_snippet = result.content[:300] + "..." if len(result.content) > 300 else result.content
            
            response_results.append(SearchResultResponse(
                document_id=result.document_id,
                url=result.url,
                title=result.title,
                description=result.description,
                content_snippet=content_snippet,
                content_type=result.content_type,
                language=result.language,
                quality_score=result.quality_score,
                final_score=result.final_score,
                crawl_time=result.crawl_time
            ))
        
        logger.info(f"Image search completed", 
                   filename=image.filename,
                   objects_detected=len(analysis_result.objects) if analysis_result.objects else 0,
                   results_count=len(response_results))
        
        return SearchResponse(
            query=f"Image search: {search_query[:100]}",
            total_results=len(response_results),
            search_time_ms=0.0,  # TODO: Add timing
            results=response_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image search service error")

@router.get("/suggestions")
async def get_search_suggestions_endpoint(
    query: str = Query(..., min_length=1, description="Partial query for suggestions"),
    limit: int = Query(default=5, ge=1, le=10, description="Maximum number of suggestions")
):
    """
    Get search suggestions based on partial query
    """
    try:
        suggestions = await get_search_suggestions(query, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        return {"suggestions": []}

@router.get("/trending")
async def get_trending_topics_endpoint(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze"),
    limit: int = Query(default=10, ge=1, le=20, description="Maximum number of topics")
):
    """
    Get trending search topics
    """
    try:
        indexer = await get_semantic_indexer()
        trending = await indexer.get_trending_topics(days=days, top_k=limit)
        return {"trending_topics": trending}
    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        return {"trending_topics": []}

@router.get("/stats")
async def get_search_stats():
    """
    Get search engine statistics
    """
    try:
        indexer = await get_semantic_indexer()
        stats = await indexer.get_statistics()
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve statistics")

# Helper functions
async def get_search_suggestions(query: str, limit: int = 5) -> List[str]:
    """Get search suggestions based on query"""
    # Placeholder implementation - in production, this would use ML models
    # or query completion from Elasticsearch
    suggestions = [
        f"{query} tutorial",
        f"{query} guide",
        f"{query} examples",
        f"{query} documentation",
        f"{query} best practices"
    ]
    return suggestions[:limit]

async def get_trending_topics(limit: int = 10) -> List[str]:
    """Get trending search topics"""
    # Placeholder implementation
    trending = [
        "machine learning",
        "artificial intelligence",
        "web development",
        "data science",
        "cloud computing"
    ]
    return trending[:limit]