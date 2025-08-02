"""
Quality Assessor for AstraFind
Handles content quality assessment and scoring
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
from textstat import textstat
import structlog

from utils.config import settings

logger = structlog.get_logger(__name__)

class QualityAssessor:
    """Assesses content quality using multiple factors"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the quality assessor"""
        if self.initialized:
            return
        
        # Load any required models or resources
        self.initialized = True
        logger.info("Quality Assessor initialized")
    
    async def assess_content_quality(
        self,
        content: str,
        title: str = "",
        url: str = "",
        language: str = "en"
    ) -> Dict[str, float]:
        """
        Assess content quality using multiple factors
        
        Returns:
            Dict with quality scores for different aspects
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Calculate various quality metrics
            readability_score = self._calculate_readability(content)
            structure_score = self._assess_structure(content, title)
            originality_score = self._assess_originality(content)
            completeness_score = self._assess_completeness(content)
            authority_score = self._assess_authority(url, content)
            freshness_score = self._assess_freshness(content)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score({
                'readability': readability_score,
                'structure': structure_score,
                'originality': originality_score,
                'completeness': completeness_score,
                'authority': authority_score,
                'freshness': freshness_score
            })
            
            return {
                'overall_quality': overall_score,
                'readability': readability_score,
                'structure': structure_score,
                'originality': originality_score,
                'completeness': completeness_score,
                'authority': authority_score,
                'freshness': freshness_score
            }
            
        except Exception as e:
            logger.error(f"Failed to assess content quality: {e}")
            return {
                'overall_quality': 0.5,
                'readability': 0.5,
                'structure': 0.5,
                'originality': 0.5,
                'completeness': 0.5,
                'authority': 0.5,
                'freshness': 0.5
            }
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        try:
            if not content.strip():
                return 0.0
            
            # Calculate Flesch Reading Ease
            flesch_score = textstat.flesch_reading_ease(content)
            
            # Normalize to 0-1 scale (Flesch typically ranges from 0-100)
            # Higher Flesch scores indicate easier reading
            normalized_score = max(0.0, min(1.0, flesch_score / 100.0))
            
            # Adjust for optimal readability (scores around 60-80 are ideal)
            if 60 <= flesch_score <= 80:
                normalized_score = min(1.0, normalized_score * 1.2)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Failed to calculate readability: {e}")
            return 0.5
    
    def _assess_structure(self, content: str, title: str = "") -> float:
        """Assess content structure and organization"""
        try:
            if not content.strip():
                return 0.0
            
            score = 0.0
            factors = 0
            
            # Check for paragraphs
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                score += 0.2
            factors += 1
            
            # Check for headings (markdown-style or HTML-like)
            heading_pattern = r'^#{1,6}\s+|^<h[1-6]>|<h[1-6][^>]*>'
            headings = re.findall(heading_pattern, content, re.MULTILINE)
            if headings:
                score += 0.2
            factors += 1
            
            # Check for lists
            list_pattern = r'^\s*[-*+]\s+|^\s*\d+\.\s+|<ul>|<ol>'
            lists = re.findall(list_pattern, content, re.MULTILINE)
            if lists:
                score += 0.15
            factors += 1
            
            # Check for links
            link_pattern = r'\[([^\]]+)\]\([^)]+\)|<a\s+href='
            links = re.findall(link_pattern, content)
            if links:
                score += 0.15
            factors += 1
            
            # Check content length (not too short, not too long)
            word_count = len(content.split())
            if 100 <= word_count <= 2000:
                score += 0.3
            elif 50 <= word_count < 100 or 2000 < word_count <= 5000:
                score += 0.2
            factors += 1
            
            # Check for title presence and relevance
            if title and len(title.strip()) > 0:
                score += 0.1
            factors += 1
            
            return score / factors if factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to assess structure: {e}")
            return 0.5
    
    def _assess_originality(self, content: str) -> float:
        """Assess content originality (basic implementation)"""
        try:
            if not content.strip():
                return 0.0
            
            # Simple heuristics for originality assessment
            score = 0.5  # Base score
            
            # Check for common boilerplate text
            boilerplate_patterns = [
                r'copyright\s+\d{4}',
                r'all\s+rights\s+reserved',
                r'privacy\s+policy',
                r'terms\s+of\s+service',
                r'cookie\s+policy'
            ]
            
            boilerplate_count = 0
            for pattern in boilerplate_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    boilerplate_count += 1
            
            # Reduce score for boilerplate content
            if boilerplate_count > 0:
                score -= min(0.3, boilerplate_count * 0.1)
            
            # Check for unique content length
            word_count = len(content.split())
            if word_count > 500:
                score += 0.2
            elif word_count > 200:
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Failed to assess originality: {e}")
            return 0.5
    
    def _assess_completeness(self, content: str) -> float:
        """Assess content completeness"""
        try:
            if not content.strip():
                return 0.0
            
            score = 0.0
            factors = 0
            
            # Check content length
            word_count = len(content.split())
            if word_count >= 100:
                score += 0.3
            elif word_count >= 50:
                score += 0.2
            factors += 1
            
            # Check for multiple sentences
            sentences = re.split(r'[.!?]+', content)
            if len(sentences) > 3:
                score += 0.2
            factors += 1
            
            # Check for different types of content (questions, statements, etc.)
            question_count = len(re.findall(r'\?', content))
            if question_count > 0:
                score += 0.1
            factors += 1
            
            # Check for numbers and data
            number_count = len(re.findall(r'\d+', content))
            if number_count > 0:
                score += 0.1
            factors += 1
            
            # Check for proper punctuation
            punctuation_count = len(re.findall(r'[.!?,;:]', content))
            if punctuation_count > 0:
                score += 0.1
            factors += 1
            
            return score / factors if factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to assess completeness: {e}")
            return 0.5
    
    def _assess_authority(self, url: str, content: str) -> float:
        """Assess content authority based on URL and content"""
        try:
            score = 0.5  # Base score
            
            # Check for authoritative domains
            authoritative_domains = [
                'wikipedia.org', 'edu', 'gov', 'org',
                'researchgate.net', 'arxiv.org', 'ieee.org',
                'acm.org', 'springer.com', 'sciencedirect.com'
            ]
            
            if any(domain in url.lower() for domain in authoritative_domains):
                score += 0.3
            
            # Check for citations or references
            citation_patterns = [
                r'\[\d+\]', r'\(\d{4}\)', r'cited', r'reference',
                r'doi\.org', r'https?://[^\s]+'
            ]
            
            citation_count = 0
            for pattern in citation_patterns:
                citation_count += len(re.findall(pattern, content, re.IGNORECASE))
            
            if citation_count > 0:
                score += min(0.2, citation_count * 0.05)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Failed to assess authority: {e}")
            return 0.5
    
    def _assess_freshness(self, content: str) -> float:
        """Assess content freshness (basic implementation)"""
        try:
            # This is a simplified implementation
            # In a real system, you'd check publication dates, last modified dates, etc.
            
            # Check for recent years mentioned
            current_year = datetime.now().year
            year_pattern = r'\b(20[12]\d)\b'
            years = re.findall(year_pattern, content)
            
            if years:
                most_recent = max(int(year) for year in years)
                years_old = current_year - most_recent
                
                if years_old <= 1:
                    return 1.0
                elif years_old <= 3:
                    return 0.8
                elif years_old <= 5:
                    return 0.6
                else:
                    return 0.4
            
            # Default score for content without clear dates
            return 0.5
            
        except Exception as e:
            logger.error(f"Failed to assess freshness: {e}")
            return 0.5
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall quality score from individual scores"""
        try:
            # Weighted average of individual scores
            weights = {
                'readability': 0.15,
                'structure': 0.20,
                'originality': 0.25,
                'completeness': 0.20,
                'authority': 0.15,
                'freshness': 0.05
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor, weight in weights.items():
                if factor in scores:
                    weighted_sum += scores[factor] * weight
                    total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Failed to calculate overall score: {e}")
            return 0.5
    
    async def get_quality_breakdown(self, content: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """Get detailed quality assessment breakdown"""
        scores = await self.assess_content_quality(content, title, url)
        
        return {
            'scores': scores,
            'recommendations': self._generate_recommendations(scores),
            'summary': self._generate_summary(scores)
        }
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        if scores.get('readability', 1.0) < 0.6:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        if scores.get('structure', 1.0) < 0.6:
            recommendations.append("Improve structure by adding headings, paragraphs, and lists")
        
        if scores.get('originality', 1.0) < 0.6:
            recommendations.append("Add more original content and reduce boilerplate text")
        
        if scores.get('completeness', 1.0) < 0.6:
            recommendations.append("Add more detailed information and examples")
        
        if scores.get('authority', 1.0) < 0.6:
            recommendations.append("Add citations and references to authoritative sources")
        
        return recommendations
    
    def _generate_summary(self, scores: Dict[str, float]) -> str:
        """Generate quality summary"""
        overall = scores.get('overall_quality', 0.5)
        
        if overall >= 0.8:
            return "High quality content"
        elif overall >= 0.6:
            return "Good quality content"
        elif overall >= 0.4:
            return "Average quality content"
        else:
            return "Low quality content"