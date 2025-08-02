import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
from sentence_transformers import SentenceTransformer
import spacy
import structlog
from ...utils.config import settings

logger = structlog.get_logger(__name__)

@dataclass
class MisinformationAnalysis:
    """Result of misinformation analysis"""
    is_misinformation: bool
    confidence_score: float
    bias_score: float
    bias_type: Optional[str] = None
    harmful_content_score: float = 0.0
    factual_accuracy_score: float = 1.0
    emotional_manipulation_score: float = 0.0
    sources_credibility: float = 1.0
    reasoning: List[str] = None
    flags: List[str] = None

@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""
    has_bias: bool
    bias_types: List[str]
    bias_scores: Dict[str, float]
    overall_bias_score: float
    political_leaning: Optional[str] = None
    gender_bias: float = 0.0
    racial_bias: float = 0.0
    cultural_bias: float = 0.0

class MisinformationFilter:
    """
    Advanced misinformation filtering and bias detection system
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models for different tasks
        self.fact_check_model = None
        self.bias_detection_model = None
        self.toxicity_model = None
        self.sentiment_model = None
        self.emotion_model = None
        
        # Sentence transformer for semantic analysis
        self.sentence_model = None
        
        # NLP models for different languages
        self.nlp_models = {}
        
        # Known patterns and indicators
        self.misinformation_patterns = [
            r'\b(fake news|hoax|conspiracy|debunked|false claim)\b',
            r'\b(miracle cure|instant fix|amazing secret)\b',
            r'\b(they don\'t want you to know|hidden truth|cover[- ]?up)\b',
            r'\b(100% effective|guaranteed|never fails)\b',
            r'\b(doctors hate|big pharma|mainstream media lies)\b'
        ]
        
        self.bias_indicators = {
            'political': [
                r'\b(liberal|conservative|leftist|right-wing|socialist|capitalist)\b',
                r'\b(democrat|republican|progressive|traditional)\b'
            ],
            'gender': [
                r'\b(typical woman|men are|women should|gender roles)\b',
                r'\b(feminine|masculine|breadwinner|housewife)\b'
            ],
            'racial': [
                r'\b(race|ethnicity|cultural superiority|minority)\b',
                r'\b(stereotyp|prejudic|discriminat)\b'
            ]
        }
        
        # Credible source domains (whitelist)
        self.credible_sources = {
            'news': [
                'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
                'cnn.com', 'nytimes.com', 'washingtonpost.com'
            ],
            'academic': [
                'ncbi.nlm.nih.gov', 'scholar.google.com', 'jstor.org',
                'springer.com', 'nature.com', 'science.org'
            ],
            'government': [
                'gov.uk', '.gov', 'who.int', 'cdc.gov', 'fda.gov'
            ]
        }
        
        # Known unreliable domains (blacklist)
        self.unreliable_sources = [
            'fakenews.com', 'conspiracy.org', 'hoax.net'
        ]
    
    async def initialize(self):
        """Initialize all ML models and NLP processors"""
        logger.info("Initializing misinformation filter...")
        
        try:
            # Initialize fact-checking model
            self.fact_check_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Replace with actual fact-check model
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize bias detection model
            self.bias_detection_tokenizer = AutoTokenizer.from_pretrained(
                "unitary/toxic-bert"
            )
            self.bias_detection_model = AutoModelForSequenceClassification.from_pretrained(
                "unitary/toxic-bert"
            ).to(self.device).eval()
            
            # Initialize toxicity detection
            self.toxicity_model = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize sentiment analysis
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize emotion detection
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                device=self.device
            )
            
            # Initialize NLP models for different languages
            await self._initialize_nlp_models()
            
            logger.info("Misinformation filter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize misinformation filter: {e}")
            raise
    
    async def _initialize_nlp_models(self):
        """Initialize spaCy models for different languages"""
        languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ar']
        
        for lang in languages:
            try:
                if lang == 'en':
                    self.nlp_models[lang] = spacy.load('en_core_web_sm')
                elif lang == 'es':
                    self.nlp_models[lang] = spacy.load('es_core_news_sm')
                elif lang == 'fr':
                    self.nlp_models[lang] = spacy.load('fr_core_news_sm')
                elif lang == 'de':
                    self.nlp_models[lang] = spacy.load('de_core_news_sm')
                else:
                    # Use multilingual model as fallback
                    self.nlp_models[lang] = spacy.load('xx_core_web_sm')
            except OSError:
                logger.warning(f"spaCy model for {lang} not found, using English model")
                self.nlp_models[lang] = self.nlp_models.get('en')
    
    async def analyze_content(
        self,
        text: str,
        url: str = None,
        title: str = None,
        language: str = 'en',
        context: Dict[str, Any] = None
    ) -> MisinformationAnalysis:
        """
        Comprehensive analysis of content for misinformation
        """
        logger.info(f"Analyzing content for misinformation: {url}")
        
        try:
            # Run multiple analyses in parallel
            results = await asyncio.gather(
                self._check_factual_accuracy(text, language),
                self._detect_bias(text, language),
                self._analyze_emotional_manipulation(text, language),
                self._check_source_credibility(url),
                self._pattern_analysis(text),
                self._linguistic_analysis(text, language),
                return_exceptions=True
            )
            
            factual_score, fact_reasoning = results[0] if not isinstance(results[0], Exception) else (0.5, [])
            bias_result = results[1] if not isinstance(results[1], Exception) else BiasDetectionResult(False, [], {}, 0.0)
            emotion_score, emotion_flags = results[2] if not isinstance(results[2], Exception) else (0.0, [])
            source_score = results[3] if not isinstance(results[3], Exception) else 0.5
            pattern_flags = results[4] if not isinstance(results[4], Exception) else []
            linguistic_flags = results[5] if not isinstance(results[5], Exception) else []
            
            # Calculate overall misinformation score
            misinformation_score = self._calculate_misinformation_score(
                factual_score, bias_result.overall_bias_score,
                emotion_score, source_score
            )
            
            # Determine if content is misinformation
            is_misinformation = misinformation_score > settings.MISINFORMATION_THRESHOLD
            
            # Combine all flags and reasoning
            all_flags = pattern_flags + linguistic_flags + emotion_flags
            all_reasoning = fact_reasoning
            
            return MisinformationAnalysis(
                is_misinformation=is_misinformation,
                confidence_score=misinformation_score,
                bias_score=bias_result.overall_bias_score,
                bias_type=bias_result.bias_types[0] if bias_result.bias_types else None,
                harmful_content_score=emotion_score,
                factual_accuracy_score=factual_score,
                emotional_manipulation_score=emotion_score,
                sources_credibility=source_score,
                reasoning=all_reasoning,
                flags=all_flags
            )
            
        except Exception as e:
            logger.error(f"Error analyzing content for misinformation: {e}")
            # Return safe default
            return MisinformationAnalysis(
                is_misinformation=False,
                confidence_score=0.5,
                bias_score=0.0,
                factual_accuracy_score=0.5,
                reasoning=["Analysis failed due to technical error"]
            )
    
    async def _check_factual_accuracy(
        self,
        text: str,
        language: str = 'en'
    ) -> Tuple[float, List[str]]:
        """Check factual accuracy using ML models"""
        reasoning = []
        
        try:
            # Use fact-checking model
            fact_result = self.fact_check_model(text[:512])  # Limit input length
            
            if isinstance(fact_result, list) and len(fact_result) > 0:
                result = fact_result[0]
                if result['label'] == 'RELIABLE':
                    score = result['score']
                    reasoning.append(f"Content appears factually reliable (confidence: {score:.2f})")
                else:
                    score = 1.0 - result['score']
                    reasoning.append(f"Content may contain factual inaccuracies (confidence: {1-score:.2f})")
            else:
                score = 0.5
                reasoning.append("Unable to determine factual accuracy")
            
            # Additional heuristics
            if self._contains_absolute_claims(text):
                score -= 0.1
                reasoning.append("Contains absolute claims that may be unverifiable")
            
            if self._lacks_sources(text):
                score -= 0.1
                reasoning.append("Content lacks credible source citations")
            
            return max(0.0, min(1.0, score)), reasoning
            
        except Exception as e:
            logger.error(f"Error in factual accuracy check: {e}")
            return 0.5, ["Factual accuracy analysis failed"]
    
    async def _detect_bias(
        self,
        text: str,
        language: str = 'en'
    ) -> BiasDetectionResult:
        """Detect various types of bias in content"""
        try:
            bias_scores = {}
            detected_types = []
            
            # Political bias detection
            political_score = await self._detect_political_bias(text)
            bias_scores['political'] = political_score
            if political_score > 0.3:
                detected_types.append('political')
            
            # Gender bias detection
            gender_score = await self._detect_gender_bias(text)
            bias_scores['gender'] = gender_score
            if gender_score > 0.3:
                detected_types.append('gender')
            
            # Racial/cultural bias detection
            racial_score = await self._detect_racial_bias(text)
            bias_scores['racial'] = racial_score
            if racial_score > 0.3:
                detected_types.append('racial')
            
            # Overall bias score
            overall_score = np.mean(list(bias_scores.values()))
            
            return BiasDetectionResult(
                has_bias=overall_score > 0.3,
                bias_types=detected_types,
                bias_scores=bias_scores,
                overall_bias_score=overall_score,
                gender_bias=gender_score,
                racial_bias=racial_score
            )
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            return BiasDetectionResult(False, [], {}, 0.0)
    
    async def _detect_political_bias(self, text: str) -> float:
        """Detect political bias using pattern matching and ML"""
        score = 0.0
        
        # Pattern-based detection
        for pattern in self.bias_indicators['political']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.1
        
        # Use toxicity model as proxy for bias detection
        try:
            inputs = self.bias_detection_tokenizer(
                text[:512], return_tensors="pt", truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bias_detection_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # Assuming the model outputs toxicity scores that correlate with bias
                bias_prob = probs[0][1].item()  # Toxic class probability
                score += bias_prob * 0.5
        except Exception as e:
            logger.warning(f"ML bias detection failed: {e}")
        
        return min(1.0, score)
    
    async def _detect_gender_bias(self, text: str) -> float:
        """Detect gender bias"""
        score = 0.0
        
        for pattern in self.bias_indicators['gender']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.15
        
        return min(1.0, score)
    
    async def _detect_racial_bias(self, text: str) -> float:
        """Detect racial/cultural bias"""
        score = 0.0
        
        for pattern in self.bias_indicators['racial']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.15
        
        return min(1.0, score)
    
    async def _analyze_emotional_manipulation(
        self,
        text: str,
        language: str = 'en'
    ) -> Tuple[float, List[str]]:
        """Analyze content for emotional manipulation"""
        flags = []
        score = 0.0
        
        try:
            # Emotion analysis
            emotions = self.emotion_model(text[:512])
            if emotions:
                # Check for manipulative emotions
                for emotion in emotions:
                    if emotion['label'] in ['anger', 'fear', 'disgust'] and emotion['score'] > 0.7:
                        score += 0.2
                        flags.append(f"High {emotion['label']} content detected")
            
            # Sentiment analysis
            sentiment = self.sentiment_model(text[:512])
            if sentiment and sentiment[0]['label'] == 'NEGATIVE' and sentiment[0]['score'] > 0.8:
                score += 0.1
                flags.append("Highly negative sentiment")
            
            # Pattern-based detection
            manipulation_patterns = [
                r'\b(urgent|immediate|act now|limited time)\b',
                r'\b(shocking|outrageous|unbelievable)\b',
                r'\b(everyone knows|nobody tells you|secret)\b'
            ]
            
            for pattern in manipulation_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.1
                    flags.append(f"Manipulative language pattern detected")
            
            return min(1.0, score), flags
            
        except Exception as e:
            logger.error(f"Error in emotional manipulation analysis: {e}")
            return 0.0, []
    
    async def _check_source_credibility(self, url: str) -> float:
        """Check the credibility of the source URL"""
        if not url:
            return 0.5
        
        try:
            domain = url.split('/')[2].lower()
            
            # Check against credible sources
            for category, sources in self.credible_sources.items():
                for source in sources:
                    if source in domain:
                        return 0.9
            
            # Check against unreliable sources
            for unreliable in self.unreliable_sources:
                if unreliable in domain:
                    return 0.1
            
            # Check domain characteristics
            score = 0.5
            
            # Government domains are more credible
            if domain.endswith('.gov') or domain.endswith('.edu'):
                score += 0.3
            
            # Check for suspicious characteristics
            if len(domain.split('.')) > 3:  # Multiple subdomains
                score -= 0.1
            
            if any(char.isdigit() for char in domain):  # Numbers in domain
                score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error checking source credibility: {e}")
            return 0.5
    
    async def _pattern_analysis(self, text: str) -> List[str]:
        """Analyze text for misinformation patterns"""
        flags = []
        
        for pattern in self.misinformation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"Misinformation pattern detected: {pattern}")
        
        return flags
    
    async def _linguistic_analysis(self, text: str, language: str) -> List[str]:
        """Perform linguistic analysis for suspicious patterns"""
        flags = []
        
        try:
            nlp = self.nlp_models.get(language, self.nlp_models['en'])
            doc = nlp(text[:1000])  # Limit processing
            
            # Check for excessive capitalization
            caps_ratio = sum(1 for char in text if char.isupper()) / len(text)
            if caps_ratio > 0.3:
                flags.append("Excessive capitalization detected")
            
            # Check for excessive punctuation
            punct_ratio = sum(1 for token in doc if token.is_punct) / len(doc)
            if punct_ratio > 0.2:
                flags.append("Excessive punctuation usage")
            
            # Check for repetitive phrases
            sentences = [sent.text for sent in doc.sents]
            if len(sentences) != len(set(sentences)):
                flags.append("Repetitive content detected")
            
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {e}")
        
        return flags
    
    def _calculate_misinformation_score(
        self,
        factual_score: float,
        bias_score: float,
        emotion_score: float,
        source_score: float
    ) -> float:
        """Calculate overall misinformation probability"""
        # Weighted combination of scores
        weights = {
            'factual': 0.4,      # Factual accuracy is most important
            'bias': 0.2,         # Bias detection
            'emotion': 0.2,      # Emotional manipulation
            'source': 0.2        # Source credibility
        }
        
        # Invert factual and source scores (higher = better)
        factual_risk = 1.0 - factual_score
        source_risk = 1.0 - source_score
        
        misinformation_score = (
            weights['factual'] * factual_risk +
            weights['bias'] * bias_score +
            weights['emotion'] * emotion_score +
            weights['source'] * source_risk
        )
        
        return max(0.0, min(1.0, misinformation_score))
    
    def _contains_absolute_claims(self, text: str) -> bool:
        """Check for absolute claims that are often inaccurate"""
        absolute_patterns = [
            r'\b(always|never|all|none|every|no one|everyone)\b',
            r'\b(100%|completely|totally|absolutely|definitely)\b'
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _lacks_sources(self, text: str) -> bool:
        """Check if content lacks source citations"""
        source_patterns = [
            r'according to',
            r'study shows',
            r'research indicates',
            r'experts say',
            r'http[s]?://',
            r'doi:',
            r'published in'
        ]
        
        for pattern in source_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get misinformation filter statistics"""
        return {
            "models_loaded": {
                "fact_check": self.fact_check_model is not None,
                "bias_detection": self.bias_detection_model is not None,
                "toxicity": self.toxicity_model is not None,
                "sentiment": self.sentiment_model is not None,
                "emotion": self.emotion_model is not None
            },
            "nlp_models": list(self.nlp_models.keys()),
            "credible_sources": sum(len(sources) for sources in self.credible_sources.values()),
            "unreliable_sources": len(self.unreliable_sources)
        }
    
    async def health_check(self) -> bool:
        """Check if the misinformation filter is healthy"""
        try:
            # Test with simple text
            test_result = await self.analyze_content(
                "This is a test sentence for health check.",
                language='en'
            )
            return test_result is not None
        except Exception as e:
            logger.error(f"Misinformation filter health check failed: {e}")
            return False