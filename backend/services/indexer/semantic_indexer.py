"""
Semantic Indexer with Multilingual NLP and Vector Embeddings
Handles document processing, embedding generation, and vector storage
"""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    pipeline, Pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.de import German
from spacy.lang.zh import Chinese
from spacy.lang.ja import Japanese
from spacy.lang.ru import Russian
import faiss
from langdetect import detect, DetectorFactory
import structlog

from ...utils.config import settings
from ..elasticsearch_client import ElasticsearchManager
from ..ml.content_scorer import ContentScorer

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = structlog.get_logger(__name__)


@dataclass
class DocumentEmbedding:
    """Represents a document with its embeddings and metadata"""
    document_id: str
    url: str
    title: str
    content: str
    language: str
    content_type: str
    
    # Embeddings
    title_embedding: Optional[np.ndarray] = None
    content_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    
    # NLP features
    entities: List[Dict[str, Any]] = None
    keywords: List[str] = None
    topics: List[Dict[str, float]] = None
    sentiment: Optional[Dict[str, float]] = None
    
    # Metadata
    word_count: int = 0
    embedding_model: str = ""
    processed_at: Optional[datetime] = None
    quality_score: float = 0.0


@dataclass
class SemanticQuery:
    """Represents a semantic search query"""
    text: str
    language: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    entities: List[str] = None
    keywords: List[str] = None
    semantic_fields: List[str] = None  # Fields to search semantically


class SemanticIndexer:
    """
    Advanced semantic indexer with multilingual support
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.sentence_model = None
        self.multilingual_model = None
        self.tokenizer = None
        
        # NLP pipelines for different languages
        self.nlp_models = {}
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.topic_pipeline = None
        
        # Vector indices
        self.title_index = None
        self.content_index = None
        self.combined_index = None
        
        # Document storage
        self.documents = {}  # document_id -> DocumentEmbedding
        self.url_to_id = {}  # url -> document_id
        
        # Configuration
        self.embedding_dim = 384  # Sentence transformer dimension
        self.max_content_length = 2048
        self.batch_size = 32
        
        # Language support
        self.supported_languages = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm', 
            'fr': 'fr_core_news_sm',
            'de': 'de_core_news_sm',
            'zh': 'zh_core_web_sm',
            'ja': 'ja_core_news_sm',
            'ru': 'ru_core_news_sm',
        }
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all models and indices"""
        try:
            logger.info("Initializing semantic indexer...")
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                device=self.device
            )
            
            # Load multilingual BERT for cross-lingual understanding
            self.tokenizer = AutoTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
            self.multilingual_model = AutoModel.from_pretrained(settings.BERT_MODEL_NAME)
            self.multilingual_model.to(self.device)
            self.multilingual_model.eval()
            
            # Initialize NLP pipelines
            await self._initialize_nlp_models()
            
            # Initialize FAISS indices
            self._initialize_vector_indices()
            
            # Load existing documents if available
            await self._load_existing_documents()
            
            self.initialized = True
            logger.info("Semantic indexer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic indexer: {e}")
            raise
    
    async def _initialize_nlp_models(self):
        """Initialize NLP models for different languages"""
        try:
            # Initialize spaCy models for supported languages
            for lang_code, model_name in self.supported_languages.items():
                try:
                    if lang_code == 'en':
                        self.nlp_models[lang_code] = English()
                    elif lang_code == 'es':
                        self.nlp_models[lang_code] = Spanish()
                    elif lang_code == 'fr':
                        self.nlp_models[lang_code] = French()
                    elif lang_code == 'de':
                        self.nlp_models[lang_code] = German()
                    elif lang_code == 'zh':
                        self.nlp_models[lang_code] = Chinese()
                    elif lang_code == 'ja':
                        self.nlp_models[lang_code] = Japanese()
                    elif lang_code == 'ru':
                        self.nlp_models[lang_code] = Russian()
                        
                    # Add pipeline components
                    if self.nlp_models[lang_code]:
                        if not self.nlp_models[lang_code].has_pipe('sentencizer'):
                            self.nlp_models[lang_code].add_pipe('sentencizer')
                            
                except Exception as e:
                    logger.warning(f"Could not load spaCy model for {lang_code}: {e}")
            
            # Initialize transformer pipelines
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("NLP models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            # Continue without advanced NLP features
    
    def _initialize_vector_indices(self):
        """Initialize FAISS vector indices"""
        try:
            # Create FAISS indices for different embedding types
            self.title_index = faiss.IndexFlatIP(self.embedding_dim)
            self.content_index = faiss.IndexFlatIP(self.embedding_dim)
            self.combined_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.title_index = faiss.index_cpu_to_gpu(res, 0, self.title_index)
                self.content_index = faiss.index_cpu_to_gpu(res, 0, self.content_index)
                self.combined_index = faiss.index_cpu_to_gpu(res, 0, self.combined_index)
            
            logger.info("Vector indices initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vector indices: {e}")
            raise
    
    async def _load_existing_documents(self):
        """Load existing documents from storage"""
        try:
            index_path = Path(settings.ML_MODELS_PATH) / "semantic_index"
            if index_path.exists():
                # Load documents metadata
                docs_file = index_path / "documents.pkl"
                if docs_file.exists():
                    with open(docs_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    
                    # Rebuild URL mapping
                    self.url_to_id = {
                        doc.url: doc_id 
                        for doc_id, doc in self.documents.items()
                    }
                
                # Load FAISS indices
                title_index_file = index_path / "title.index"
                if title_index_file.exists():
                    self.title_index = faiss.read_index(str(title_index_file))
                
                content_index_file = index_path / "content.index"
                if content_index_file.exists():
                    self.content_index = faiss.read_index(str(content_index_file))
                
                combined_index_file = index_path / "combined.index"
                if combined_index_file.exists():
                    self.combined_index = faiss.read_index(str(combined_index_file))
                
                logger.info(f"Loaded {len(self.documents)} existing documents")
            
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
    
    async def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document with semantic processing
        
        Args:
            document: Document data from crawler
            
        Returns:
            Success status
        """
        if not self.initialized:
            raise RuntimeError("Indexer not initialized")
        
        try:
            url = document.get('url', '')
            if url in self.url_to_id:
                logger.debug(f"Document already indexed: {url}")
                return True
            
            # Create document embedding
            doc_embedding = await self._process_document(document)
            
            if not doc_embedding:
                logger.warning(f"Failed to process document: {url}")
                return False
            
            # Generate embeddings
            await self._generate_embeddings(doc_embedding)
            
            # Extract NLP features
            await self._extract_nlp_features(doc_embedding)
            
            # Add to vector indices
            self._add_to_indices(doc_embedding)
            
            # Store document
            self.documents[doc_embedding.document_id] = doc_embedding
            self.url_to_id[doc_embedding.url] = doc_embedding.document_id
            
            # Index in Elasticsearch for hybrid search
            await self._index_in_elasticsearch(doc_embedding)
            
            logger.debug(f"Successfully indexed document: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {document.get('url', 'unknown')}: {e}")
            return False
    
    async def _process_document(self, document: Dict[str, Any]) -> Optional[DocumentEmbedding]:
        """Process raw document into DocumentEmbedding"""
        try:
            # Extract basic information
            url = document.get('url', '')
            title = document.get('title', '')
            content_data = document.get('content', {})
            content = content_data.get('text', '')
            language = document.get('language', 'unknown')
            content_type = document.get('content_type', 'general')
            quality_score = document.get('content_score', 0.0)
            
            # Generate document ID
            doc_id = hashlib.sha256(url.encode()).hexdigest()
            
            # Detect language if unknown
            if language == 'unknown' and content:
                try:
                    language = detect(content[:1000])
                except:
                    language = 'en'  # Default to English
            
            # Truncate content if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
            
            return DocumentEmbedding(
                document_id=doc_id,
                url=url,
                title=title,
                content=content,
                language=language,
                content_type=content_type,
                word_count=len(content.split()),
                quality_score=quality_score,
                processed_at=datetime.utcnow(),
                embedding_model=settings.SENTENCE_TRANSFORMER_MODEL
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    async def _generate_embeddings(self, doc: DocumentEmbedding):
        """Generate embeddings for document components"""
        try:
            # Generate title embedding
            if doc.title:
                doc.title_embedding = self.sentence_model.encode([doc.title])[0]
            
            # Generate content embedding
            if doc.content:
                # For long content, use sliding window approach
                content_chunks = self._chunk_text(doc.content, chunk_size=512)
                chunk_embeddings = self.sentence_model.encode(content_chunks)
                
                # Average embeddings for final content representation
                doc.content_embedding = np.mean(chunk_embeddings, axis=0)
            
            # Generate combined embedding
            combined_text = f"{doc.title} {doc.content}".strip()
            if combined_text:
                doc.combined_embedding = self.sentence_model.encode([combined_text])[0]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks if chunks else [text]
    
    async def _extract_nlp_features(self, doc: DocumentEmbedding):
        """Extract NLP features like entities, keywords, sentiment"""
        try:
            text = f"{doc.title} {doc.content}".strip()
            if not text:
                return
            
            # Named Entity Recognition
            if self.ner_pipeline:
                try:
                    entities = self.ner_pipeline(text[:512])  # Limit length for NER
                    doc.entities = [
                        {
                            'text': ent['word'],
                            'label': ent['entity_group'],
                            'confidence': ent['score']
                        }
                        for ent in entities
                        if ent['score'] > 0.8  # High confidence only
                    ]
                except Exception as e:
                    logger.debug(f"NER failed: {e}")
                    doc.entities = []
            
            # Sentiment Analysis
            if self.sentiment_pipeline:
                try:
                    sentiment_result = self.sentiment_pipeline(text[:512])
                    if sentiment_result:
                        doc.sentiment = {
                            'label': sentiment_result[0]['label'],
                            'score': sentiment_result[0]['score']
                        }
                except Exception as e:
                    logger.debug(f"Sentiment analysis failed: {e}")
            
            # Keyword extraction using spaCy
            if doc.language in self.nlp_models:
                try:
                    nlp = self.nlp_models[doc.language]
                    processed_doc = nlp(text[:1000])  # Limit for performance
                    
                    # Extract keywords (nouns, proper nouns, adjectives)
                    keywords = []
                    for token in processed_doc:
                        if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                            len(token.text) > 2 and 
                            not token.is_stop and 
                            not token.is_punct):
                            keywords.append(token.lemma_.lower())
                    
                    # Remove duplicates and take top keywords
                    doc.keywords = list(set(keywords))[:20]
                    
                except Exception as e:
                    logger.debug(f"Keyword extraction failed: {e}")
                    doc.keywords = []
            
        except Exception as e:
            logger.error(f"Error extracting NLP features: {e}")
    
    def _add_to_indices(self, doc: DocumentEmbedding):
        """Add document embeddings to FAISS indices"""
        try:
            if doc.title_embedding is not None:
                # Normalize embedding for cosine similarity
                title_emb = doc.title_embedding / np.linalg.norm(doc.title_embedding)
                self.title_index.add(title_emb.reshape(1, -1).astype('float32'))
            
            if doc.content_embedding is not None:
                content_emb = doc.content_embedding / np.linalg.norm(doc.content_embedding)
                self.content_index.add(content_emb.reshape(1, -1).astype('float32'))
            
            if doc.combined_embedding is not None:
                combined_emb = doc.combined_embedding / np.linalg.norm(doc.combined_embedding)
                self.combined_index.add(combined_emb.reshape(1, -1).astype('float32'))
                
        except Exception as e:
            logger.error(f"Error adding to indices: {e}")
    
    async def _index_in_elasticsearch(self, doc: DocumentEmbedding):
        """Index document in Elasticsearch for hybrid search"""
        try:
            es_doc = {
                'url': doc.url,
                'title': doc.title,
                'content': doc.content,
                'language': doc.language,
                'content_type': doc.content_type,
                'word_count': doc.word_count,
                'quality_score': doc.quality_score,
                'entities': doc.entities or [],
                'keywords': doc.keywords or [],
                'sentiment': doc.sentiment,
                'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
                'embedding_model': doc.embedding_model,
            }
            
            # Add embeddings as dense vectors (if Elasticsearch supports it)
            if doc.combined_embedding is not None:
                es_doc['combined_embedding'] = doc.combined_embedding.tolist()
            
            await ElasticsearchManager.index_document(
                index_name=f"{settings.ELASTICSEARCH_INDEX_PREFIX}-documents",
                doc_id=doc.document_id,
                document=es_doc
            )
            
        except Exception as e:
            logger.error(f"Error indexing in Elasticsearch: {e}")
    
    async def semantic_search(
        self, 
        query: Union[str, SemanticQuery], 
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[DocumentEmbedding, float]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query string or SemanticQuery object
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.initialized:
            raise RuntimeError("Indexer not initialized")
        
        try:
            # Process query
            if isinstance(query, str):
                query = SemanticQuery(text=query)
            
            # Generate query embedding
            if query.embedding is None:
                query.embedding = self.sentence_model.encode([query.text])[0]
                query.embedding = query.embedding / np.linalg.norm(query.embedding)
            
            # Search in combined index
            query_vector = query.embedding.reshape(1, -1).astype('float32')
            similarities, indices = self.combined_index.search(query_vector, min(top_k * 2, len(self.documents)))
            
            # Get documents and filter by threshold
            results = []
            doc_list = list(self.documents.values())
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(doc_list) and similarity >= similarity_threshold:
                    results.append((doc_list[idx], float(similarity)))
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def find_similar_documents(
        self, 
        document_id: str, 
        top_k: int = 5
    ) -> List[Tuple[DocumentEmbedding, float]]:
        """Find documents similar to a given document"""
        try:
            if document_id not in self.documents:
                return []
            
            source_doc = self.documents[document_id]
            if source_doc.combined_embedding is None:
                return []
            
            # Use the document's embedding as query
            query = SemanticQuery(
                text=source_doc.title,
                embedding=source_doc.combined_embedding
            )
            
            results = await self.semantic_search(query, top_k + 1)  # +1 to exclude self
            
            # Remove the source document from results
            filtered_results = [
                (doc, score) for doc, score in results 
                if doc.document_id != document_id
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    async def get_document_clusters(self, num_clusters: int = 10) -> Dict[int, List[str]]:
        """Cluster documents using K-means on embeddings"""
        try:
            if len(self.documents) < num_clusters:
                return {}
            
            # Collect all embeddings
            embeddings = []
            doc_ids = []
            
            for doc_id, doc in self.documents.items():
                if doc.combined_embedding is not None:
                    embeddings.append(doc.combined_embedding)
                    doc_ids.append(doc_id)
            
            if len(embeddings) < num_clusters:
                return {}
            
            embeddings = np.array(embeddings)
            
            # Use FAISS for K-means clustering
            kmeans = faiss.Kmeans(
                embeddings.shape[1], 
                num_clusters, 
                niter=20, 
                verbose=False
            )
            kmeans.train(embeddings.astype('float32'))
            
            # Get cluster assignments
            _, cluster_assignments = kmeans.index.search(embeddings.astype('float32'), 1)
            
            # Group documents by cluster
            clusters = {}
            for doc_id, cluster_id in zip(doc_ids, cluster_assignments.flatten()):
                cluster_id = int(cluster_id)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(doc_id)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering documents: {e}")
            return {}
    
    async def get_trending_topics(self, days: int = 7, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics based on recent documents"""
        try:
            from collections import Counter
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Collect keywords from recent documents
            all_keywords = []
            all_entities = []
            
            for doc in self.documents.values():
                if doc.processed_at and doc.processed_at >= cutoff_date:
                    if doc.keywords:
                        all_keywords.extend(doc.keywords)
                    if doc.entities:
                        all_entities.extend([ent['text'].lower() for ent in doc.entities])
            
            # Count frequencies
            keyword_counts = Counter(all_keywords)
            entity_counts = Counter(all_entities)
            
            # Combine and rank
            trending_topics = []
            
            # Top keywords
            for keyword, count in keyword_counts.most_common(top_k):
                trending_topics.append({
                    'text': keyword,
                    'type': 'keyword',
                    'count': count,
                    'score': count / len(all_keywords) if all_keywords else 0
                })
            
            # Top entities
            for entity, count in entity_counts.most_common(top_k):
                trending_topics.append({
                    'text': entity,
                    'type': 'entity',
                    'count': count,
                    'score': count / len(all_entities) if all_entities else 0
                })
            
            # Sort by score and return top_k
            trending_topics.sort(key=lambda x: x['score'], reverse=True)
            return trending_topics[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    async def save_index(self):
        """Save the semantic index to disk"""
        try:
            index_path = Path(settings.ML_MODELS_PATH) / "semantic_index"
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            docs_file = index_path / "documents.pkl"
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save FAISS indices
            if self.title_index.ntotal > 0:
                faiss.write_index(self.title_index, str(index_path / "title.index"))
            
            if self.content_index.ntotal > 0:
                faiss.write_index(self.content_index, str(index_path / "content.index"))
            
            if self.combined_index.ntotal > 0:
                faiss.write_index(self.combined_index, str(index_path / "combined.index"))
            
            # Save metadata
            metadata = {
                'total_documents': len(self.documents),
                'embedding_model': settings.SENTENCE_TRANSFORMER_MODEL,
                'saved_at': datetime.utcnow().isoformat(),
                'supported_languages': list(self.supported_languages.keys())
            }
            
            with open(index_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Semantic index saved with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        try:
            language_stats = {}
            content_type_stats = {}
            quality_stats = {'total': 0, 'sum': 0.0}
            
            for doc in self.documents.values():
                # Language statistics
                lang = doc.language or 'unknown'
                language_stats[lang] = language_stats.get(lang, 0) + 1
                
                # Content type statistics
                content_type = doc.content_type or 'unknown'
                content_type_stats[content_type] = content_type_stats.get(content_type, 0) + 1
                
                # Quality statistics
                quality_stats['total'] += 1
                quality_stats['sum'] += doc.quality_score
            
            avg_quality = quality_stats['sum'] / quality_stats['total'] if quality_stats['total'] > 0 else 0
            
            return {
                'total_documents': len(self.documents),
                'total_urls': len(self.url_to_id),
                'language_distribution': language_stats,
                'content_type_distribution': content_type_stats,
                'average_quality_score': avg_quality,
                'supported_languages': list(self.supported_languages.keys()),
                'embedding_model': settings.SENTENCE_TRANSFORMER_MODEL,
                'index_sizes': {
                    'title_index': self.title_index.ntotal if self.title_index else 0,
                    'content_index': self.content_index.ntotal if self.content_index else 0,
                    'combined_index': self.combined_index.ntotal if self.combined_index else 0,
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check if the indexer is healthy"""
        try:
            if not self.initialized:
                return False
            
            # Test embedding generation
            test_text = "This is a test document for health check."
            test_embedding = self.sentence_model.encode([test_text])
            
            return len(test_embedding) > 0 and test_embedding.shape[1] == self.embedding_dim
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False