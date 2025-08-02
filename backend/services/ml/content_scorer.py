"""
AI-Powered Content Scoring System
Uses machine learning models to evaluate content quality
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import structlog

from ...utils.config import settings


logger = structlog.get_logger(__name__)


class ContentScorer:
    """
    AI-powered content quality scoring system
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quality_model = None
        self.sentence_model = None
        self.tokenizer = None
        self.initialized = False
        
        # Quality scoring weights
        self.scoring_weights = {
            'semantic_quality': 0.3,
            'readability': 0.2,
            'structure': 0.15,
            'originality': 0.15,
            'relevance': 0.1,
            'trustworthiness': 0.1,
        }
        
        # Content quality patterns
        self.quality_indicators = {
            'positive': [
                r'\b(research|study|analysis|investigation)\b',
                r'\b(according to|source:|cited|reference)\b',
                r'\b(data shows|statistics|evidence)\b',
                r'\b(expert|professor|researcher|scientist)\b',
                r'\b(published|journal|academic|peer-reviewed)\b',
            ],
            'negative': [
                r'\b(click here|free|amazing|incredible|shocking)\b',
                r'\b(you won\'t believe|doctors hate|one weird trick)\b',
                r'\b(buy now|limited time|act fast|special offer)\b',
                r'\[advertisement\]|\bads?\b|\bsponsored\b',
                r'\b(fake|hoax|conspiracy|debunked)\b',
            ],
            'spam': [
                r'[A-Z]{3,}.*[A-Z]{3,}',  # Excessive caps
                r'!!!+',  # Multiple exclamation marks
                r'\$\$+',  # Money symbols
                r'www\.[a-z]+\.(com|net|org)',  # URLs in text
            ]
        }
    
    async def initialize(self):
        """Initialize ML models for content scoring"""
        try:
            logger.info("Initializing content scoring models...")
            
            # Load sentence transformer for semantic analysis
            self.sentence_model = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                device=self.device
            )
            
            # Load BERT for quality assessment
            self.tokenizer = AutoTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
            self.quality_model = AutoModel.from_pretrained(settings.BERT_MODEL_NAME)
            self.quality_model.to(self.device)
            self.quality_model.eval()
            
            self.initialized = True
            logger.info("Content scoring models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize content scoring models: {e}")
            raise
    
    def score_content(
        self,
        text: str,
        title: str = "",
        content_type: str = "general",
        word_count: int = 0,
        image_count: int = 0,
        link_count: int = 0,
        **kwargs
    ) -> float:
        """
        Calculate comprehensive content quality score
        
        Args:
            text: Main content text
            title: Page title
            content_type: Type of content (article, news, blog, etc.)
            word_count: Number of words in content
            image_count: Number of images
            link_count: Number of links
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.initialized:
            logger.warning("Content scorer not initialized, using fallback scoring")
            return self._fallback_score(text, title, word_count)
        
        try:
            scores = {}
            
            # Semantic quality score
            scores['semantic_quality'] = self._calculate_semantic_quality(text, title)
            
            # Readability score
            scores['readability'] = self._calculate_readability(text)
            
            # Structure score
            scores['structure'] = self._calculate_structure_score(
                text, title, word_count, image_count, link_count
            )
            
            # Originality score
            scores['originality'] = self._calculate_originality_score(text)
            
            # Relevance score
            scores['relevance'] = self._calculate_relevance_score(text, title, content_type)
            
            # Trustworthiness score
            scores['trustworthiness'] = self._calculate_trustworthiness_score(text)
            
            # Calculate weighted final score
            final_score = sum(
                scores[component] * self.scoring_weights[component]
                for component in scores
            )
            
            # Apply content type modifiers
            final_score = self._apply_content_type_modifiers(final_score, content_type)
            
            # Ensure score is within bounds
            final_score = max(0.0, min(1.0, final_score))
            
            logger.debug(f"Content scoring breakdown: {scores}, final: {final_score}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in content scoring: {e}")
            return self._fallback_score(text, title, word_count)
    
    def _calculate_semantic_quality(self, text: str, title: str) -> float:
        """Calculate semantic quality using transformer models"""
        try:
            # Combine title and text for analysis
            combined_text = f"{title} {text}"[:512]  # Limit for BERT
            
            # Tokenize and encode
            inputs = self.tokenizer(
                combined_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.quality_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Calculate semantic coherence
            # This is a simplified approach - in production, you'd use a trained model
            coherence_score = self._calculate_coherence_from_embeddings(embeddings)
            
            # Get sentence embeddings for diversity analysis
            sentences = self._split_into_sentences(text)
            if len(sentences) > 1:
                sentence_embeddings = self.sentence_model.encode(sentences[:10])  # Limit sentences
                diversity_score = self._calculate_semantic_diversity(sentence_embeddings)
            else:
                diversity_score = 0.5
            
            # Combine coherence and diversity
            semantic_score = (coherence_score * 0.7) + (diversity_score * 0.3)
            
            return semantic_score
            
        except Exception as e:
            logger.error(f"Error calculating semantic quality: {e}")
            return 0.5
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using multiple metrics"""
        try:
            if len(text.strip()) < 50:
                return 0.3  # Very short text gets low readability
            
            # Basic readability metrics
            words = text.split()
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return 0.3
            
            # Average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Average syllables per word (approximation)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            # Flesch Reading Ease approximation
            flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            readability_score = max(0, min(100, flesch_score)) / 100
            
            # Adjust for very short or very long sentences
            if avg_words_per_sentence < 8:
                readability_score *= 0.8  # Too short sentences
            elif avg_words_per_sentence > 25:
                readability_score *= 0.7  # Too long sentences
            
            return readability_score
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.5
    
    def _calculate_structure_score(
        self, text: str, title: str, word_count: int, image_count: int, link_count: int
    ) -> float:
        """Calculate content structure quality score"""
        try:
            score = 0.5  # Base score
            
            # Title quality
            if title:
                title_words = len(title.split())
                if 5 <= title_words <= 15:  # Optimal title length
                    score += 0.1
                elif title_words < 3:
                    score -= 0.1
            
            # Word count optimization
            if 300 <= word_count <= 2000:
                score += 0.2
            elif word_count < 100:
                score -= 0.3
            elif word_count > 3000:
                score -= 0.1
            
            # Paragraph structure
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                avg_paragraph_length = word_count / len(paragraphs)
                if 50 <= avg_paragraph_length <= 150:
                    score += 0.1
            
            # Image to text ratio
            if word_count > 500:
                ideal_images = word_count / 300  # Roughly 1 image per 300 words
                image_ratio = min(image_count / ideal_images, 1.0) if ideal_images > 0 else 0
                score += image_ratio * 0.1
            
            # Link density
            if word_count > 0:
                link_density = link_count / word_count
                if 0.01 <= link_density <= 0.05:  # 1-5% link density is good
                    score += 0.05
                elif link_density > 0.1:  # Too many links
                    score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {e}")
            return 0.5
    
    def _calculate_originality_score(self, text: str) -> float:
        """Calculate content originality score"""
        try:
            # Check for common copied patterns
            originality_score = 1.0
            
            # Penalize excessive repetition
            words = text.lower().split()
            if len(words) > 50:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.4:  # Too much repetition
                    originality_score -= 0.3
            
            # Check for boilerplate text patterns
            boilerplate_patterns = [
                r'copyright.*all rights reserved',
                r'terms of service',
                r'privacy policy',
                r'cookies? policy',
                r'lorem ipsum',
            ]
            
            text_lower = text.lower()
            for pattern in boilerplate_patterns:
                if re.search(pattern, text_lower):
                    originality_score -= 0.1
            
            # Check for excessive quotes (might indicate copied content)
            quote_count = text.count('"') + text.count("'")
            if len(text) > 0:
                quote_density = quote_count / len(text)
                if quote_density > 0.1:  # More than 10% quotes
                    originality_score -= 0.2
            
            return max(0.0, min(1.0, originality_score))
            
        except Exception as e:
            logger.error(f"Error calculating originality score: {e}")
            return 0.7
    
    def _calculate_relevance_score(self, text: str, title: str, content_type: str) -> float:
        """Calculate content relevance score"""
        try:
            relevance_score = 0.5
            
            # Title-content alignment
            if title and text:
                title_words = set(title.lower().split())
                text_words = set(text.lower().split())
                
                # Remove common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                title_words -= stop_words
                text_words -= stop_words
                
                if title_words and text_words:
                    overlap = len(title_words & text_words) / len(title_words)
                    relevance_score += overlap * 0.3
            
            # Content type specific keywords
            type_keywords = {
                'news': ['reported', 'according', 'source', 'breaking', 'update'],
                'article': ['introduction', 'conclusion', 'analysis', 'research', 'study'],
                'blog': ['opinion', 'think', 'believe', 'experience', 'personal'],
                'academic': ['methodology', 'results', 'discussion', 'conclusion', 'abstract'],
            }
            
            if content_type in type_keywords:
                text_lower = text.lower()
                keyword_count = sum(1 for keyword in type_keywords[content_type] if keyword in text_lower)
                relevance_score += (keyword_count / len(type_keywords[content_type])) * 0.2
            
            return max(0.0, min(1.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    def _calculate_trustworthiness_score(self, text: str) -> float:
        """Calculate content trustworthiness score"""
        try:
            trust_score = 0.5
            text_lower = text.lower()
            
            # Positive indicators
            for pattern in self.quality_indicators['positive']:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                trust_score += min(matches * 0.02, 0.1)  # Cap at 0.1 per category
            
            # Negative indicators
            for pattern in self.quality_indicators['negative']:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                trust_score -= min(matches * 0.05, 0.2)  # Penalize more heavily
            
            # Spam indicators
            for pattern in self.quality_indicators['spam']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                trust_score -= min(matches * 0.1, 0.3)
            
            # Check for balanced perspective (good sign)
            perspective_words = ['however', 'although', 'despite', 'nevertheless', 'on the other hand']
            perspective_count = sum(1 for word in perspective_words if word in text_lower)
            if perspective_count > 0:
                trust_score += min(perspective_count * 0.02, 0.1)
            
            return max(0.0, min(1.0, trust_score))
            
        except Exception as e:
            logger.error(f"Error calculating trustworthiness score: {e}")
            return 0.5
    
    def _apply_content_type_modifiers(self, score: float, content_type: str) -> float:
        """Apply content type specific score modifiers"""
        modifiers = {
            'news': 1.0,
            'article': 1.1,  # Slight boost for articles
            'academic': 1.2,  # Higher value for academic content
            'blog': 0.9,     # Slight reduction for blog posts
            'social': 0.7,   # Lower value for social media content
            'product': 0.8,  # Commercial content gets lower score
            'general': 1.0,
        }
        
        modifier = modifiers.get(content_type, 1.0)
        return min(1.0, score * modifier)
    
    def _fallback_score(self, text: str, title: str, word_count: int) -> float:
        """Fallback scoring when ML models are not available"""
        score = 0.5  # Base score
        
        # Simple heuristics
        if 100 <= word_count <= 2000:
            score += 0.2
        
        if title and len(title.split()) >= 3:
            score += 0.1
        
        # Check for obvious spam patterns
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in ['buy now', 'click here', 'free money']):
            score -= 0.3
        
        return max(0.1, min(1.0, score))
    
    def _calculate_coherence_from_embeddings(self, embeddings: torch.Tensor) -> float:
        """Calculate coherence score from embeddings"""
        try:
            # Simple coherence measure based on embedding norms
            norms = torch.norm(embeddings, dim=1)
            avg_norm = norms.mean().item()
            
            # Normalize to 0-1 range (this is a simplification)
            coherence = min(1.0, avg_norm / 10.0)
            return coherence
            
        except Exception:
            return 0.5
    
    def _calculate_semantic_diversity(self, sentence_embeddings: np.ndarray) -> float:
        """Calculate semantic diversity between sentences"""
        try:
            if len(sentence_embeddings) < 2:
                return 0.5
            
            # Calculate pairwise cosine similarities
            similarities = []
            for i in range(len(sentence_embeddings)):
                for j in range(i + 1, len(sentence_embeddings)):
                    cos_sim = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                        np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j])
                    )
                    similarities.append(cos_sim)
            
            # Lower average similarity means higher diversity
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
            
            return max(0.0, min(1.0, diversity))
            
        except Exception:
            return 0.5
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least one syllable