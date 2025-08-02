"""
Hybrid Search Ranking System
Combines traditional BM25 with deep learning models (BERT, T5) for optimal search results
"""

import asyncio
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import structlog

from ...utils.config import settings


logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with all ranking components"""
    document_id: str
    url: str
    title: str
    content: str
    description: str
    content_type: str
    language: str
    
    # Ranking scores
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    click_score: float = 0.0
    personalization_score: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    crawl_time: Optional[datetime] = None
    word_count: int = 0
    image_count: int = 0
    link_count: int = 0


@dataclass
class SearchQuery:
    """Represents a search query with context"""
    text: str
    user_id: Optional[str] = None
    location: Optional[str] = None
    language: str = "en"
    content_type_filter: Optional[str] = None
    time_filter: Optional[str] = None  # "day", "week", "month", "year"
    safe_search: bool = True
    personalized: bool = True


class HybridSearchRanker:
    """
    Hybrid search ranking system combining multiple signals
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentence_transformer = None
        self.t5_model = None
        self.t5_tokenizer = None
        
        # Vector index for semantic search
        self.semantic_index = None
        self.document_embeddings = {}
        
        # BM25 parameters
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        
        # Ranking weights
        self.ranking_weights = {
            'bm25': 0.30,           # Traditional keyword matching
            'semantic': 0.25,       # Semantic similarity
            'quality': 0.20,        # Content quality score
            'freshness': 0.10,      # Recency of content
            'click_through': 0.10,  # User engagement signals
            'personalization': 0.05, # User preference matching
        }
        
        # Caches
        self.query_cache = {}
        self.embedding_cache = {}
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all ML models and indices"""
        try:
            logger.info("Initializing hybrid search ranker...")
            
            # Load BERT for query understanding
            self.bert_tokenizer = AutoTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
            self.bert_model = AutoModel.from_pretrained(settings.BERT_MODEL_NAME)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # Load sentence transformer for semantic search
            self.sentence_transformer = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                device=self.device
            )
            
            # Load T5 for query expansion
            self.t5_tokenizer = T5Tokenizer.from_pretrained(settings.T5_MODEL_NAME)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(settings.T5_MODEL_NAME)
            self.t5_model.to(self.device)
            self.t5_model.eval()
            
            # Initialize FAISS index for semantic search
            self.semantic_index = faiss.IndexFlatIP(384)  # Sentence transformer dimension
            if torch.cuda.is_available():
                self.semantic_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.semantic_index
                )
            
            self.initialized = True
            logger.info("Hybrid search ranker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search ranker: {e}")
            raise
    
    async def rank_results(
        self,
        query: SearchQuery,
        candidate_documents: List[Dict[str, Any]],
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Rank search results using hybrid approach
        
        Args:
            query: Search query with context
            candidate_documents: Raw documents from Elasticsearch
            max_results: Maximum number of results to return
            
        Returns:
            Ranked list of search results
        """
        if not self.initialized:
            raise RuntimeError("Ranker not initialized")
        
        if not candidate_documents:
            return []
        
        try:
            # Convert documents to SearchResult objects
            search_results = [
                self._document_to_search_result(doc) for doc in candidate_documents
            ]
            
            # Expand query for better matching
            expanded_query = await self._expand_query(query.text)
            
            # Calculate various ranking signals
            await asyncio.gather(
                self._calculate_bm25_scores(expanded_query, search_results),
                self._calculate_semantic_scores(query.text, search_results),
                self._calculate_quality_scores(search_results),
                self._calculate_freshness_scores(search_results),
                self._calculate_click_scores(search_results),
                self._calculate_personalization_scores(query, search_results),
            )
            
            # Combine scores and rank
            self._calculate_final_scores(search_results)
            
            # Sort by final score
            search_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply post-processing filters
            filtered_results = self._post_process_results(query, search_results)
            
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error ranking search results: {e}")
            return search_results[:max_results]  # Return unranked results as fallback
    
    async def _expand_query(self, query: str) -> str:
        """Expand query using T5 for better matching"""
        try:
            # Check cache first
            if query in self.query_cache:
                return self.query_cache[query]
            
            # Prepare input for T5
            input_text = f"expand query: {query}"
            input_ids = self.t5_tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            ).to(self.device)
            
            # Generate expanded query
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.7
                )
            
            expanded = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Combine original and expanded query
            expanded_query = f"{query} {expanded}"
            
            # Cache result
            self.query_cache[query] = expanded_query
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query  # Return original query on error
    
    async def _calculate_bm25_scores(self, query: str, results: List[SearchResult]):
        """Calculate BM25 scores for all results"""
        try:
            query_terms = query.lower().split()
            
            # Calculate document frequency for each term
            N = len(results)  # Total number of documents
            df = {}  # Document frequency
            
            for result in results:
                doc_text = f"{result.title} {result.content} {result.description}".lower()
                doc_terms = set(doc_text.split())
                
                for term in query_terms:
                    if term in doc_terms:
                        df[term] = df.get(term, 0) + 1
            
            # Calculate BM25 score for each document
            for result in results:
                doc_text = f"{result.title} {result.content} {result.description}".lower()
                doc_terms = doc_text.split()
                doc_length = len(doc_terms)
                
                # Calculate average document length (simplified)
                avg_doc_length = sum(len(r.content.split()) for r in results) / len(results)
                
                score = 0.0
                for term in query_terms:
                    if term in doc_terms:
                        # Term frequency in document
                        tf = doc_terms.count(term)
                        
                        # Inverse document frequency
                        idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                        
                        # BM25 formula
                        score += idf * (tf * (self.bm25_k1 + 1)) / (
                            tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_length / avg_doc_length)
                        )
                
                result.bm25_score = max(0.0, score)
                
        except Exception as e:
            logger.error(f"Error calculating BM25 scores: {e}")
    
    async def _calculate_semantic_scores(self, query: str, results: List[SearchResult]):
        """Calculate semantic similarity scores using sentence transformers"""
        try:
            # Get query embedding
            query_embedding = self.sentence_transformer.encode([query])[0]
            
            # Calculate similarity with each document
            for result in results:
                # Combine title and content for embedding
                doc_text = f"{result.title} {result.content[:500]}"  # Limit length
                
                # Check cache first
                doc_id = result.document_id
                if doc_id in self.embedding_cache:
                    doc_embedding = self.embedding_cache[doc_id]
                else:
                    doc_embedding = self.sentence_transformer.encode([doc_text])[0]
                    self.embedding_cache[doc_id] = doc_embedding
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                result.semantic_score = max(0.0, similarity)
                
        except Exception as e:
            logger.error(f"Error calculating semantic scores: {e}")
    
    async def _calculate_quality_scores(self, results: List[SearchResult]):
        """Use pre-calculated quality scores from crawling"""
        try:
            for result in results:
                # Quality score should already be calculated during crawling
                # This is a placeholder for any additional quality adjustments
                result.quality_score = getattr(result, 'quality_score', 0.5)
                
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
    
    async def _calculate_freshness_scores(self, results: List[SearchResult]):
        """Calculate freshness scores based on content age"""
        try:
            now = datetime.utcnow()
            
            for result in results:
                if result.crawl_time:
                    age_days = (now - result.crawl_time).days
                    
                    # Exponential decay for freshness
                    if age_days <= 1:
                        freshness = 1.0
                    elif age_days <= 7:
                        freshness = 0.8
                    elif age_days <= 30:
                        freshness = 0.6
                    elif age_days <= 90:
                        freshness = 0.4
                    elif age_days <= 365:
                        freshness = 0.2
                    else:
                        freshness = 0.1
                    
                    result.freshness_score = freshness
                else:
                    result.freshness_score = 0.5  # Default for unknown age
                    
        except Exception as e:
            logger.error(f"Error calculating freshness scores: {e}")
    
    async def _calculate_click_scores(self, results: List[SearchResult]):
        """Calculate scores based on historical click-through rates"""
        try:
            # This would typically query a database of click statistics
            # For now, we'll use a placeholder implementation
            
            for result in results:
                # Placeholder: In production, this would fetch real CTR data
                # Higher quality content types get higher base CTR
                base_ctr = {
                    'news': 0.6,
                    'article': 0.7,
                    'academic': 0.8,
                    'blog': 0.5,
                    'social': 0.4,
                    'product': 0.3,
                }.get(result.content_type, 0.5)
                
                result.click_score = base_ctr
                
        except Exception as e:
            logger.error(f"Error calculating click scores: {e}")
    
    async def _calculate_personalization_scores(self, query: SearchQuery, results: List[SearchResult]):
        """Calculate personalization scores based on user preferences"""
        try:
            if not query.personalized or not query.user_id:
                # No personalization
                for result in results:
                    result.personalization_score = 0.5
                return
            
            # This would typically query user preference data
            # For now, we'll use a simplified implementation
            
            # Placeholder user preferences (in production, fetch from database)
            user_preferences = {
                'preferred_content_types': ['article', 'news'],
                'preferred_languages': ['en'],
                'interests': ['technology', 'science', 'programming'],
            }
            
            for result in results:
                score = 0.5  # Base score
                
                # Content type preference
                if result.content_type in user_preferences.get('preferred_content_types', []):
                    score += 0.2
                
                # Language preference
                if result.language in user_preferences.get('preferred_languages', []):
                    score += 0.1
                
                # Interest matching (simplified keyword matching)
                content_lower = f"{result.title} {result.description}".lower()
                interest_matches = sum(
                    1 for interest in user_preferences.get('interests', [])
                    if interest in content_lower
                )
                score += min(interest_matches * 0.1, 0.2)
                
                result.personalization_score = min(1.0, score)
                
        except Exception as e:
            logger.error(f"Error calculating personalization scores: {e}")
    
    def _calculate_final_scores(self, results: List[SearchResult]):
        """Calculate final weighted scores"""
        try:
            # Normalize scores to 0-1 range
            self._normalize_scores(results)
            
            for result in results:
                final_score = (
                    result.bm25_score * self.ranking_weights['bm25'] +
                    result.semantic_score * self.ranking_weights['semantic'] +
                    result.quality_score * self.ranking_weights['quality'] +
                    result.freshness_score * self.ranking_weights['freshness'] +
                    result.click_score * self.ranking_weights['click_through'] +
                    result.personalization_score * self.ranking_weights['personalization']
                )
                
                result.final_score = final_score
                
        except Exception as e:
            logger.error(f"Error calculating final scores: {e}")
    
    def _normalize_scores(self, results: List[SearchResult]):
        """Normalize all scores to 0-1 range"""
        try:
            score_types = ['bm25_score', 'semantic_score', 'quality_score', 
                          'freshness_score', 'click_score', 'personalization_score']
            
            for score_type in score_types:
                scores = [getattr(result, score_type) for result in results]
                if not scores:
                    continue
                
                min_score = min(scores)
                max_score = max(scores)
                
                if max_score > min_score:
                    for result in results:
                        old_score = getattr(result, score_type)
                        normalized = (old_score - min_score) / (max_score - min_score)
                        setattr(result, score_type, normalized)
                        
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
    
    def _post_process_results(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchResult]:
        """Apply post-processing filters to results"""
        try:
            filtered_results = results
            
            # Language filtering
            if query.language:
                filtered_results = [
                    r for r in filtered_results 
                    if r.language == query.language or r.language == 'unknown'
                ]
            
            # Content type filtering
            if query.content_type_filter:
                filtered_results = [
                    r for r in filtered_results 
                    if r.content_type == query.content_type_filter
                ]
            
            # Time filtering
            if query.time_filter and query.time_filter != 'all':
                cutoff_days = {
                    'day': 1,
                    'week': 7,
                    'month': 30,
                    'year': 365,
                }.get(query.time_filter, 365)
                
                cutoff_date = datetime.utcnow() - timedelta(days=cutoff_days)
                filtered_results = [
                    r for r in filtered_results 
                    if r.crawl_time and r.crawl_time >= cutoff_date
                ]
            
            # Safe search filtering (placeholder)
            if query.safe_search:
                # In production, this would filter out adult content
                pass
            
            # Diversity enhancement (avoid too many results from same domain)
            filtered_results = self._enhance_diversity(filtered_results)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return results
    
    def _enhance_diversity(self, results: List[SearchResult], max_per_domain: int = 3) -> List[SearchResult]:
        """Enhance result diversity by limiting results per domain"""
        try:
            domain_counts = {}
            diverse_results = []
            
            for result in results:
                # Extract domain from URL
                domain = self._extract_domain(result.url)
                current_count = domain_counts.get(domain, 0)
                
                if current_count < max_per_domain:
                    diverse_results.append(result)
                    domain_counts[domain] = current_count + 1
            
            return diverse_results
            
        except Exception as e:
            logger.error(f"Error enhancing diversity: {e}")
            return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def _document_to_search_result(self, doc: Dict[str, Any]) -> SearchResult:
        """Convert Elasticsearch document to SearchResult"""
        return SearchResult(
            document_id=doc.get('_id', ''),
            url=doc.get('_source', {}).get('url', ''),
            title=doc.get('_source', {}).get('title', ''),
            content=doc.get('_source', {}).get('content', {}).get('text', ''),
            description=doc.get('_source', {}).get('description', ''),
            content_type=doc.get('_source', {}).get('content_type', 'general'),
            language=doc.get('_source', {}).get('language', 'unknown'),
            crawl_time=doc.get('_source', {}).get('crawl_time'),
            word_count=doc.get('_source', {}).get('content', {}).get('word_count', 0),
            image_count=len(doc.get('_source', {}).get('content', {}).get('images', [])),
            link_count=len(doc.get('_source', {}).get('content', {}).get('links', [])),
            quality_score=doc.get('_source', {}).get('content_score', 0.5),
        )
    
    async def health_check(self) -> bool:
        """Check if the ranker is healthy and operational"""
        try:
            if not self.initialized:
                return False
            
            # Test a simple ranking operation
            test_query = SearchQuery(text="test query")
            test_docs = [{
                '_id': 'test',
                '_source': {
                    'url': 'https://example.com',
                    'title': 'Test Document',
                    'content': {'text': 'This is a test document'},
                    'description': 'Test description',
                    'content_type': 'article',
                    'language': 'en',
                    'content_score': 0.8,
                }
            }]
            
            results = await self.rank_results(test_query, test_docs)
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False