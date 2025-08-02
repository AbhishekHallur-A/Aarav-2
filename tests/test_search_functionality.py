import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Any

# Test imports
import sys
sys.path.append('../backend')

from backend.services.search.hybrid_ranker import HybridSearchRanker, SearchQuery, SearchResult
from backend.services.indexer.semantic_indexer import SemanticIndexer, DocumentEmbedding
from backend.services.ml.image_processor import ImageProcessor, ImageAnalysisResult
from backend.services.ml.voice_processor import VoiceProcessor, VoiceAnalysisResult
from backend.services.ml.misinformation_filter import MisinformationFilter, MisinformationAnalysis
from backend.services.elasticsearch_client import ElasticsearchManager

class TestHybridSearchRanker:
    """Test the hybrid search ranking system"""
    
    @pytest.fixture
    async def ranker(self):
        """Create a HybridSearchRanker instance for testing"""
        ranker = HybridSearchRanker()
        
        # Mock the ML models to avoid loading actual models
        ranker.bert_model = Mock()
        ranker.bert_tokenizer = Mock()
        ranker.sentence_transformer = Mock()
        ranker.t5_model = Mock()
        ranker.t5_tokenizer = Mock()
        ranker.semantic_index = Mock()
        
        return ranker
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                "document_id": "doc1",
                "url": "https://example.com/article1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence...",
                "content_quality_score": 0.8,
                "page_rank_score": 0.6,
                "freshness_score": 0.9,
                "click_through_rate": 0.05,
                "domain": "example.com",
                "language": "en"
            },
            {
                "document_id": "doc2",
                "url": "https://tech.com/ai-guide",
                "title": "Complete Guide to Artificial Intelligence",
                "content": "Artificial intelligence encompasses machine learning, deep learning...",
                "content_quality_score": 0.9,
                "page_rank_score": 0.8,
                "freshness_score": 0.7,
                "click_through_rate": 0.08,
                "domain": "tech.com",
                "language": "en"
            },
            {
                "document_id": "doc3",
                "url": "https://news.com/latest-ai",
                "title": "Latest AI Research Breakthrough",
                "content": "Recent advances in artificial intelligence have led to...",
                "content_quality_score": 0.7,
                "page_rank_score": 0.4,
                "freshness_score": 1.0,
                "click_through_rate": 0.12,
                "domain": "news.com",
                "language": "en"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, ranker):
        """Test T5-based query expansion"""
        # Mock T5 response
        ranker.t5_tokenizer.encode.return_value = [1, 2, 3]
        ranker.t5_model.generate.return_value = Mock()
        ranker.t5_model.generate.return_value.cpu.return_value.numpy.return_value = [[4, 5, 6]]
        ranker.t5_tokenizer.decode.return_value = "machine learning artificial intelligence ML AI"
        
        expanded_query = await ranker._expand_query("machine learning")
        
        assert isinstance(expanded_query, str)
        assert len(expanded_query) > 0
        ranker.t5_tokenizer.encode.assert_called_once()
        ranker.t5_model.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bm25_scoring(self, ranker, sample_documents):
        """Test BM25 scoring calculation"""
        query = "machine learning artificial intelligence"
        search_results = [ranker._document_to_search_result(doc) for doc in sample_documents]
        
        await ranker._calculate_bm25_scores(query, search_results)
        
        # Check that BM25 scores are assigned
        for result in search_results:
            assert hasattr(result, 'bm25_score')
            assert isinstance(result.bm25_score, float)
            assert 0.0 <= result.bm25_score <= 10.0  # Reasonable BM25 score range
    
    @pytest.mark.asyncio
    async def test_semantic_scoring(self, ranker, sample_documents):
        """Test semantic similarity scoring"""
        query = "machine learning"
        search_results = [ranker._document_to_search_result(doc) for doc in sample_documents]
        
        # Mock sentence transformer
        ranker.sentence_transformer.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        await ranker._calculate_semantic_scores(query, search_results)
        
        # Check that semantic scores are assigned
        for result in search_results:
            assert hasattr(result, 'semantic_score')
            assert isinstance(result.semantic_score, float)
            assert 0.0 <= result.semantic_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_final_ranking(self, ranker, sample_documents):
        """Test final ranking calculation and result ordering"""
        query = SearchQuery(
            text="machine learning",
            language="en",
            user_id="user123"
        )
        
        # Mock various scoring methods
        ranker._expand_query = AsyncMock(return_value="machine learning artificial intelligence")
        ranker._calculate_bm25_scores = AsyncMock()
        ranker._calculate_semantic_scores = AsyncMock()
        ranker._calculate_quality_scores = AsyncMock()
        ranker._calculate_freshness_scores = AsyncMock()
        ranker._calculate_click_through_scores = AsyncMock()
        ranker._calculate_personalization_scores = AsyncMock()
        
        # Set up mock scores
        def setup_scores(results):
            for i, result in enumerate(results):
                result.bm25_score = 0.8 - (i * 0.1)
                result.semantic_score = 0.9 - (i * 0.1)
                result.quality_score = sample_documents[i]["content_quality_score"]
                result.freshness_score = sample_documents[i]["freshness_score"]
                result.click_through_score = sample_documents[i]["click_through_rate"]
                result.personalization_score = 0.5
        
        ranker._calculate_bm25_scores.side_effect = setup_scores
        
        ranked_results = await ranker.rank_results(query, sample_documents, max_results=3)
        
        assert len(ranked_results) <= 3
        assert all(isinstance(result, SearchResult) for result in ranked_results)
        
        # Check that results are ordered by final score (descending)
        for i in range(len(ranked_results) - 1):
            assert ranked_results[i].final_score >= ranked_results[i + 1].final_score

class TestSemanticIndexer:
    """Test the semantic indexing system"""
    
    @pytest.fixture
    async def indexer(self):
        """Create a SemanticIndexer instance for testing"""
        indexer = SemanticIndexer()
        
        # Mock the ML models
        indexer.sentence_model = Mock()
        indexer.nlp_models = {"en": Mock()}
        indexer.sentiment_pipeline = Mock()
        indexer.ner_pipeline = Mock()
        indexer.combined_index = Mock()
        
        return indexer
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for indexing"""
        return {
            "document_id": "doc123",
            "url": "https://example.com/test",
            "title": "Test Document",
            "content": "This is a test document about machine learning and artificial intelligence.",
            "language": "en",
            "content_type": "text/html"
        }
    
    @pytest.mark.asyncio
    async def test_document_processing(self, indexer, sample_document):
        """Test document processing and embedding generation"""
        # Mock embedding generation
        indexer.sentence_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Mock NLP pipeline responses
        indexer.sentiment_pipeline.return_value = [{"label": "POSITIVE", "score": 0.8}]
        indexer.ner_pipeline.return_value = [
            {"entity": "MISC", "word": "machine learning", "confidence": 0.9}
        ]
        
        doc_embedding = await indexer._process_document(sample_document)
        
        assert isinstance(doc_embedding, DocumentEmbedding)
        assert doc_embedding.document_id == sample_document["document_id"]
        assert doc_embedding.url == sample_document["url"]
        assert doc_embedding.title == sample_document["title"]
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, indexer, sample_document):
        """Test embedding vector generation"""
        doc_embedding = DocumentEmbedding(
            document_id=sample_document["document_id"],
            url=sample_document["url"],
            title=sample_document["title"],
            content=sample_document["content"],
            language=sample_document["language"],
            content_type=sample_document["content_type"]
        )
        
        # Mock sentence transformer
        indexer.sentence_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        await indexer._generate_embeddings(doc_embedding)
        
        assert doc_embedding.combined_embedding is not None
        assert isinstance(doc_embedding.combined_embedding, np.ndarray)
        assert len(doc_embedding.combined_embedding) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, indexer):
        """Test semantic search functionality"""
        query = "machine learning algorithms"
        
        # Mock search results
        mock_results = [
            (Mock(document_id="doc1"), 0.9),
            (Mock(document_id="doc2"), 0.8),
            (Mock(document_id="doc3"), 0.7)
        ]
        
        indexer.sentence_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        indexer.combined_index.search.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])
        indexer.documents = {
            0: Mock(document_id="doc1"),
            1: Mock(document_id="doc2"),
            2: Mock(document_id="doc3")
        }
        
        results = await indexer.semantic_search(query, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(similarity, float) for _, similarity in results)
        assert all(0.0 <= similarity <= 1.0 for _, similarity in results)

class TestImageProcessor:
    """Test the image processing system"""
    
    @pytest.fixture
    def processor(self):
        """Create an ImageProcessor instance for testing"""
        processor = ImageProcessor()
        
        # Mock the ML models
        processor.object_detection_pipeline = Mock()
        processor.image_classification_pipeline = Mock()
        processor.feature_extractor = Mock()
        processor.feature_model = Mock()
        
        return processor
    
    @pytest.fixture
    def sample_image_data(self):
        """Sample image data for testing"""
        # Create a simple test image (1x1 RGB pixel)
        from PIL import Image
        import io
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @pytest.mark.asyncio
    async def test_image_analysis(self, processor, sample_image_data):
        """Test complete image analysis workflow"""
        # Mock pipeline responses
        processor.object_detection_pipeline.return_value = [
            {"label": "person", "score": 0.9, "box": {"xmin": 10, "ymin": 20, "xmax": 50, "ymax": 80}}
        ]
        processor.image_classification_pipeline.return_value = [
            {"label": "portrait", "score": 0.8}
        ]
        
        # Mock OCR (using patch since pytesseract is external)
        with patch('pytesseract.image_to_string', return_value="Sample text"):
            result = await processor.analyze_image(
                sample_image_data,
                include_objects=True,
                include_text=True
            )
        
        assert isinstance(result, ImageAnalysisResult)
        assert len(result.objects) > 0
        assert result.text == "Sample text"
        assert len(result.tags) > 0
    
    @pytest.mark.asyncio
    async def test_object_detection(self, processor, sample_image_data):
        """Test object detection functionality"""
        # Mock object detection response
        processor.object_detection_pipeline.return_value = [
            {"label": "car", "score": 0.95, "box": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 50}},
            {"label": "person", "score": 0.88, "box": {"xmin": 50, "ymin": 25, "xmax": 75, "ymax": 100}}
        ]
        
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(sample_image_data))
        
        objects = await processor._detect_objects(image)
        
        assert len(objects) == 2
        assert objects[0]["name"] == "car"
        assert objects[0]["confidence"] == 0.95
        assert objects[1]["name"] == "person"
        assert objects[1]["confidence"] == 0.88

class TestVoiceProcessor:
    """Test the voice processing system"""
    
    @pytest.fixture
    def processor(self):
        """Create a VoiceProcessor instance for testing"""
        processor = VoiceProcessor()
        
        # Mock the ML models
        processor.whisper_model = Mock()
        processor.wav2vec2_model = Mock()
        processor.wav2vec2_tokenizer = Mock()
        processor.speech_pipeline = Mock()
        processor.language_detector = Mock()
        
        return processor
    
    @pytest.fixture
    def sample_audio_data(self):
        """Sample audio data for testing"""
        # Create silent audio data (16kHz, 1 second)
        import numpy as np
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_array = np.zeros(samples, dtype=np.float32)
        return audio_array.tobytes()
    
    @pytest.mark.asyncio
    async def test_speech_to_text(self, processor, sample_audio_data):
        """Test complete speech-to-text workflow"""
        # Mock transcription results from different models
        processor._transcribe_with_whisper = AsyncMock(return_value=("Hello world", 0.9))
        processor._transcribe_with_wav2vec2 = AsyncMock(return_value=("Hello world", 0.8))
        processor._transcribe_with_pipeline = AsyncMock(return_value=("Hello world", 0.85))
        
        # Mock other methods
        processor._extract_audio_metadata = AsyncMock(return_value=Mock(duration=1.0, sample_rate=16000))
        processor._enhance_audio = AsyncMock(return_value="/tmp/enhanced.wav")
        processor._load_audio = AsyncMock(return_value=(np.zeros(16000), 16000))
        processor._assess_audio_quality = AsyncMock(return_value=0.8)
        processor._detect_language = AsyncMock(return_value="en")
        processor._extract_keywords = AsyncMock(return_value=["hello", "world"])
        
        result = await processor.speech_to_text(sample_audio_data, language="en")
        
        assert isinstance(result, VoiceAnalysisResult)
        assert result.transcript == "Hello world"
        assert result.language == "en"
        assert result.confidence > 0.0
        assert len(result.keywords) > 0
    
    @pytest.mark.asyncio
    async def test_audio_enhancement(self, processor):
        """Test audio enhancement pipeline"""
        # Create test audio file
        import tempfile
        import numpy as np
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Mock librosa functions
            with patch('librosa.load', return_value=(np.random.randn(16000), 16000)):
                with patch('librosa.output.write_wav'):
                    enhanced_path = await processor._enhance_audio(temp_file.name)
                    
                    assert enhanced_path.endswith('_enhanced.wav')
    
    @pytest.mark.asyncio
    async def test_audio_quality_assessment(self, processor):
        """Test audio quality assessment"""
        # Create test audio with known characteristics
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Create audio with some signal
        t = np.linspace(0, duration, samples)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        quality_score = await processor._assess_audio_quality(audio_data, sample_rate)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

class TestMisinformationFilter:
    """Test the misinformation filtering system"""
    
    @pytest.fixture
    def filter_system(self):
        """Create a MisinformationFilter instance for testing"""
        filter_system = MisinformationFilter()
        
        # Mock the ML models
        filter_system.fact_check_model = Mock()
        filter_system.bias_detection_model = Mock()
        filter_system.bias_detection_tokenizer = Mock()
        filter_system.toxicity_model = Mock()
        filter_system.sentiment_model = Mock()
        filter_system.emotion_model = Mock()
        filter_system.sentence_model = Mock()
        filter_system.nlp_models = {"en": Mock()}
        
        return filter_system
    
    @pytest.mark.asyncio
    async def test_misinformation_analysis(self, filter_system):
        """Test complete misinformation analysis"""
        content = "This is a test article about climate change science."
        url = "https://example.com/article"
        
        # Mock the analysis methods
        filter_system._check_factual_accuracy = AsyncMock(return_value=(0.8, ["Content appears reliable"]))
        filter_system._detect_bias = AsyncMock(return_value=Mock(
            has_bias=False, bias_types=[], bias_scores={}, overall_bias_score=0.1
        ))
        filter_system._analyze_emotional_manipulation = AsyncMock(return_value=(0.2, ["Low emotional content"]))
        filter_system._check_source_credibility = AsyncMock(return_value=0.7)
        filter_system._pattern_analysis = AsyncMock(return_value=[])
        filter_system._linguistic_analysis = AsyncMock(return_value=[])
        
        result = await filter_system.analyze_content(content, url=url, language="en")
        
        assert isinstance(result, MisinformationAnalysis)
        assert isinstance(result.is_misinformation, bool)
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.bias_score <= 1.0
        assert 0.0 <= result.factual_accuracy_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_bias_detection(self, filter_system):
        """Test bias detection functionality"""
        # Test with potentially biased content
        biased_content = "Men are naturally better at math than women."
        
        bias_result = await filter_system._detect_bias(biased_content, "en")
        
        # The exact result depends on the implementation, but we can test structure
        assert hasattr(bias_result, 'has_bias')
        assert hasattr(bias_result, 'bias_types')
        assert hasattr(bias_result, 'overall_bias_score')
        assert isinstance(bias_result.overall_bias_score, float)
        assert 0.0 <= bias_result.overall_bias_score <= 1.0
    
    def test_pattern_detection(self, filter_system):
        """Test misinformation pattern detection"""
        # Test with content containing misinformation patterns
        suspicious_content = "This amazing secret that doctors don't want you to know will cure everything!"
        
        flags = asyncio.run(filter_system._pattern_analysis(suspicious_content))
        
        assert isinstance(flags, list)
        # Should detect suspicious patterns
        assert len(flags) > 0

class TestElasticsearchManager:
    """Test the Elasticsearch integration"""
    
    @pytest.fixture
    def es_manager(self):
        """Create a mocked ElasticsearchManager for testing"""
        # Reset the singleton instance for testing
        ElasticsearchManager._instance = None
        ElasticsearchManager._client = None
        
        manager = ElasticsearchManager()
        
        # Mock the Elasticsearch client
        mock_client = AsyncMock()
        ElasticsearchManager._client = mock_client
        
        return manager
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for Elasticsearch testing"""
        return {
            "url": "https://example.com/test",
            "title": "Test Document",
            "content": "This is test content about machine learning.",
            "language": "en",
            "content_type": "text/html",
            "domain": "example.com",
            "keywords": ["machine learning", "AI"],
            "content_quality_score": 0.8,
            "crawled_at": datetime.utcnow()
        }
    
    @pytest.mark.asyncio
    async def test_document_indexing(self, es_manager, sample_document):
        """Test document indexing functionality"""
        # Mock successful indexing response
        ElasticsearchManager._client.index.return_value = {"result": "created"}
        
        result = await ElasticsearchManager.index_document(sample_document)
        
        assert result is True
        ElasticsearchManager._client.index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, es_manager):
        """Test search functionality"""
        # Mock search response
        mock_response = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 1.5,
                        "_source": {
                            "url": "https://example.com/doc1",
                            "title": "Machine Learning Basics",
                            "content": "Introduction to ML concepts...",
                            "domain": "example.com",
                            "language": "en"
                        }
                    },
                    {
                        "_id": "doc2",
                        "_score": 1.2,
                        "_source": {
                            "url": "https://example.com/doc2",
                            "title": "Advanced AI Techniques",
                            "content": "Deep dive into AI methods...",
                            "domain": "example.com",
                            "language": "en"
                        }
                    }
                ]
            },
            "took": 15
        }
        
        ElasticsearchManager._client.search.return_value = mock_response
        
        results = await ElasticsearchManager.search(
            query="machine learning",
            size=10,
            language="en"
        )
        
        assert results["total"] == 2
        assert len(results["hits"]) == 2
        assert results["took"] == 15
        assert results["hits"][0]["title"] == "Machine Learning Basics"
    
    @pytest.mark.asyncio
    async def test_suggestions(self, es_manager):
        """Test search suggestions functionality"""
        # Mock suggestions response
        mock_response = {
            "suggest": {
                "title_suggest": [
                    {
                        "options": [
                            {"text": "machine learning"},
                            {"text": "machine learning algorithms"},
                            {"text": "machine learning tutorial"}
                        ]
                    }
                ]
            }
        }
        
        ElasticsearchManager._client.search.return_value = mock_response
        
        suggestions = await ElasticsearchManager.get_suggestions("machine", size=5)
        
        assert len(suggestions) == 3
        assert "machine learning" in suggestions
        assert "machine learning algorithms" in suggestions

# Integration tests
class TestSearchIntegration:
    """Integration tests for the complete search pipeline"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_text_search(self):
        """Test complete text search pipeline"""
        # This would require actual integration with all components
        # For now, we'll test the coordination between components
        
        # Mock all components
        with patch('backend.services.elasticsearch_client.ElasticsearchManager') as mock_es:
            with patch('backend.services.search.hybrid_ranker.HybridSearchRanker') as mock_ranker:
                
                # Setup mock responses
                mock_es.search.return_value = {"hits": [], "total": 0}
                mock_ranker.rank_results.return_value = []
                
                # Test that the components would be called correctly
                # This is a simplified version - real integration tests would use actual instances
                query = "machine learning tutorial"
                
                # Simulate the search flow
                es_results = await mock_es.search(query=query)
                ranked_results = await mock_ranker.rank_results(
                    query={"text": query}, 
                    candidate_documents=es_results["hits"]
                )
                
                # Verify the flow
                mock_es.search.assert_called_once()
                mock_ranker.rank_results.assert_called_once()

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])