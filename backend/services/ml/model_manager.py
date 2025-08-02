"""
Model Manager for AstraFind
Handles ML model loading and management
"""

import asyncio
import os
from typing import Dict, Optional, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import structlog

from ...utils.config import settings

logger = structlog.get_logger(__name__)

class ModelManager:
    """Manages ML models for AstraFind"""
    
    _models: Dict[str, Any] = {}
    _tokenizers: Dict[str, Any] = {}
    _initialized: bool = False
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize and load ML models"""
        if cls._initialized:
            return
            
        try:
            logger.info("Loading ML models...")
            
            # Create models directory if it doesn't exist
            models_path = Path(settings.ML_MODELS_PATH)
            models_path.mkdir(parents=True, exist_ok=True)
            
            # Load BERT model for text classification and embeddings
            await cls._load_bert_model()
            
            # Load T5 model for text generation and query expansion
            await cls._load_t5_model()
            
            # Load Sentence Transformer for semantic similarity
            await cls._load_sentence_transformer()
            
            cls._initialized = True
            logger.info("All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    @classmethod
    async def _load_bert_model(cls) -> None:
        """Load BERT model asynchronously"""
        try:
            logger.info(f"Loading BERT model: {settings.BERT_MODEL_NAME}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                settings.BERT_MODEL_NAME,
                cache_dir=settings.ML_MODELS_PATH
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                settings.BERT_MODEL_NAME,
                cache_dir=settings.ML_MODELS_PATH
            )
            
            # Set to evaluation mode
            model.eval()
            
            cls._tokenizers['bert'] = tokenizer
            cls._models['bert'] = model
            
            logger.info("BERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    @classmethod
    async def _load_t5_model(cls) -> None:
        """Load T5 model asynchronously"""
        try:
            logger.info(f"Loading T5 model: {settings.T5_MODEL_NAME}")
            
            # Load tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                settings.T5_MODEL_NAME,
                cache_dir=settings.ML_MODELS_PATH
            )
            
            # Load model
            model = T5ForConditionalGeneration.from_pretrained(
                settings.T5_MODEL_NAME,
                cache_dir=settings.ML_MODELS_PATH
            )
            
            # Set to evaluation mode
            model.eval()
            
            cls._tokenizers['t5'] = tokenizer
            cls._models['t5'] = model
            
            logger.info("T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            raise
    
    @classmethod
    async def _load_sentence_transformer(cls) -> None:
        """Load Sentence Transformer model asynchronously"""
        try:
            logger.info(f"Loading Sentence Transformer: {settings.SENTENCE_TRANSFORMER_MODEL}")
            
            # Load model
            model = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                cache_folder=settings.ML_MODELS_PATH
            )
            
            cls._models['sentence_transformer'] = model
            
            logger.info("Sentence Transformer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            raise
    
    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup ML models"""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cls._models.clear()
            cls._tokenizers.clear()
            cls._initialized = False
            
            logger.info("ML models cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup ML models: {e}")
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check ML models health"""
        try:
            if not cls._initialized:
                return False
            
            # Check if all required models are loaded
            required_models = ['bert', 't5', 'sentence_transformer']
            required_tokenizers = ['bert', 't5']
            
            for model_name in required_models:
                if model_name not in cls._models:
                    logger.error(f"Missing model: {model_name}")
                    return False
            
            for tokenizer_name in required_tokenizers:
                if tokenizer_name not in cls._tokenizers:
                    logger.error(f"Missing tokenizer: {tokenizer_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"ML models health check failed: {e}")
            return False
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        """Get a specific model"""
        if not cls._initialized:
            raise RuntimeError("ModelManager not initialized")
        
        return cls._models.get(model_name)
    
    @classmethod
    def get_tokenizer(cls, tokenizer_name: str) -> Optional[Any]:
        """Get a specific tokenizer"""
        if not cls._initialized:
            raise RuntimeError("ModelManager not initialized")
        
        return cls._tokenizers.get(tokenizer_name)
    
    @classmethod
    def get_bert_model(cls) -> Any:
        """Get BERT model"""
        return cls.get_model('bert')
    
    @classmethod
    def get_bert_tokenizer(cls) -> Any:
        """Get BERT tokenizer"""
        return cls.get_tokenizer('bert')
    
    @classmethod
    def get_t5_model(cls) -> Any:
        """Get T5 model"""
        return cls.get_model('t5')
    
    @classmethod
    def get_t5_tokenizer(cls) -> Any:
        """Get T5 tokenizer"""
        return cls.get_tokenizer('t5')
    
    @classmethod
    def get_sentence_transformer(cls) -> Any:
        """Get Sentence Transformer model"""
        return cls.get_model('sentence_transformer')
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if ModelManager is initialized"""
        return cls._initialized
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models"""
        return list(cls._models.keys())
    
    @classmethod
    def get_available_tokenizers(cls) -> list:
        """Get list of available tokenizers"""
        return list(cls._tokenizers.keys())