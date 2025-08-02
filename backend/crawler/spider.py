"""
AstraFind Intelligent Web Crawler
AI-powered content scoring and crawling strategies
"""

import re
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.exceptions import DropItem
import numpy as np
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import structlog

from ..services.ml.content_scorer import ContentScorer
from ..services.ml.quality_assessor import QualityAssessor
from ..services.database import DatabaseManager
from ..utils.config import settings


# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = structlog.get_logger(__name__)


class AstraFindSpider(Spider):
    """
    Intelligent web crawler with AI-based content scoring
    """
    
    name = "astrafind"
    allowed_domains = []  # Will be populated dynamically
    start_urls = []       # Will be populated from seed URLs
    
    custom_settings = {
        'USER_AGENT': settings.CRAWLER_USER_AGENT,
        'ROBOTSTXT_OBEY': settings.CRAWLER_RESPECT_ROBOTS_TXT,
        'CONCURRENT_REQUESTS': settings.CRAWLER_CONCURRENT_REQUESTS,
        'DOWNLOAD_DELAY': settings.CRAWLER_DOWNLOAD_DELAY,
        'RANDOMIZE_DOWNLOAD_DELAY': settings.CRAWLER_RANDOMIZE_DELAY,
        'COOKIES_ENABLED': True,
        'TELNETCONSOLE_ENABLED': False,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
        'ITEM_PIPELINES': {
            'astrafind.crawler.pipelines.DuplicatesPipeline': 200,
            'astrafind.crawler.pipelines.ContentScoringPipeline': 300,
            'astrafind.crawler.pipelines.QualityFilterPipeline': 400,
            'astrafind.crawler.pipelines.IndexingPipeline': 500,
        },
        'DOWNLOADER_MIDDLEWARES': {
            'astrafind.crawler.middlewares.RotateUserAgentMiddleware': 400,
            'astrafind.crawler.middlewares.ProxyMiddleware': 500,
            'astrafind.crawler.middlewares.SmartThrottleMiddleware': 600,
        },
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_scorer = ContentScorer()
        self.quality_assessor = QualityAssessor()
        self.crawled_urls: Set[str] = set()
        self.url_scores: Dict[str, float] = {}
        self.domain_stats: Dict[str, Dict[str, Any]] = {}
        self.session_start = datetime.utcnow()
        
        # Content patterns for different page types
        self.content_patterns = {
            'article': [
                r'<article[^>]*>',
                r'class="[^"]*article[^"]*"',
                r'class="[^"]*post[^"]*"',
                r'class="[^"]*content[^"]*"',
            ],
            'news': [
                r'class="[^"]*news[^"]*"',
                r'class="[^"]*story[^"]*"',
                r'<time[^>]*>',
                r'datetime=',
            ],
            'blog': [
                r'class="[^"]*blog[^"]*"',
                r'class="[^"]*entry[^"]*"',
                r'rel="author"',
            ],
            'product': [
                r'class="[^"]*product[^"]*"',
                r'class="[^"]*item[^"]*"',
                r'price',
                r'buy.*now',
            ],
            'social': [
                r'class="[^"]*tweet[^"]*"',
                r'class="[^"]*post[^"]*"',
                r'class="[^"]*status[^"]*"',
            ]
        }
    
    def start_requests(self):
        """Generate initial requests from seed URLs"""
        seed_urls = self.get_seed_urls()
        
        for url in seed_urls:
            yield Request(
                url=url,
                callback=self.parse,
                meta={
                    'depth': 0,
                    'discovery_time': datetime.utcnow(),
                    'parent_url': None,
                    'crawl_priority': 1.0,
                }
            )
    
    def get_seed_urls(self) -> List[str]:
        """Retrieve seed URLs from database or configuration"""
        # In production, this would fetch from database
        return [
            'https://example.com',
            'https://news.ycombinator.com',
            'https://www.reddit.com',
            'https://stackoverflow.com',
            'https://medium.com',
        ]
    
    def parse(self, response: Response):
        """Main parsing method with AI-powered content extraction"""
        try:
            # Extract basic page information
            url = response.url
            html_content = response.text
            
            # Skip if already crawled
            if url in self.crawled_urls:
                logger.debug(f"Skipping already crawled URL: {url}")
                return
            
            self.crawled_urls.add(url)
            
            # Extract metadata
            metadata = self.extract_metadata(response)
            
            # Extract main content
            content = self.extract_content(response)
            
            # Skip if content is too short or low quality
            if len(content.get('text', '').strip()) < 100:
                logger.debug(f"Skipping short content: {url}")
                return
            
            # Detect content type and language
            content_type = self.detect_content_type(html_content, content['text'])
            language = self.detect_language(content['text'])
            
            # Calculate AI-based content score
            content_score = self.calculate_content_score(content, metadata, content_type)
            
            # Store URL score for link prioritization
            self.url_scores[url] = content_score
            
            # Update domain statistics
            self.update_domain_stats(url, content_score, content_type)
            
            # Create page item
            page_item = {
                'url': url,
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'keywords': metadata.get('keywords', []),
                'content': content,
                'metadata': metadata,
                'content_type': content_type,
                'language': language,
                'content_score': content_score,
                'crawl_time': datetime.utcnow(),
                'depth': response.meta.get('depth', 0),
                'parent_url': response.meta.get('parent_url'),
            }
            
            yield page_item
            
            # Extract and prioritize links
            links = self.extract_links(response)
            prioritized_links = self.prioritize_links(links, content_score, url)
            
            # Generate follow-up requests
            for link_url, priority in prioritized_links:
                if self.should_follow_link(link_url, response.meta.get('depth', 0)):
                    yield Request(
                        url=link_url,
                        callback=self.parse,
                        meta={
                            'depth': response.meta.get('depth', 0) + 1,
                            'discovery_time': datetime.utcnow(),
                            'parent_url': url,
                            'crawl_priority': priority,
                        },
                        priority=int(priority * 100)  # Scrapy priority (0-1000)
                    )
            
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}", exc_info=True)
    
    def extract_metadata(self, response: Response) -> Dict[str, Any]:
        """Extract page metadata"""
        metadata = {}
        
        # Title
        title = response.css('title::text').get()
        if title:
            metadata['title'] = title.strip()
        
        # Meta tags
        meta_tags = response.css('meta')
        for meta in meta_tags:
            name = meta.css('::attr(name)').get()
            property_attr = meta.css('::attr(property)').get()
            content = meta.css('::attr(content)').get()
            
            if content:
                if name == 'description':
                    metadata['description'] = content.strip()
                elif name == 'keywords':
                    metadata['keywords'] = [k.strip() for k in content.split(',')]
                elif name == 'author':
                    metadata['author'] = content.strip()
                elif name == 'robots':
                    metadata['robots'] = content.strip()
                elif property_attr and property_attr.startswith('og:'):
                    metadata[property_attr] = content.strip()
                elif name and name.startswith('twitter:'):
                    metadata[name] = content.strip()
        
        # Canonical URL
        canonical = response.css('link[rel="canonical"]::attr(href)').get()
        if canonical:
            metadata['canonical'] = response.urljoin(canonical)
        
        # Last modified
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            metadata['last_modified'] = last_modified.decode('utf-8')
        
        # Content type
        content_type = response.headers.get('Content-Type')
        if content_type:
            metadata['content_type'] = content_type.decode('utf-8')
        
        return metadata
    
    def extract_content(self, response: Response) -> Dict[str, Any]:
        """Extract main content from page"""
        content = {}
        
        # Remove script and style elements
        cleaned_html = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_html = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract text content
        text_content = response.css('body ::text').getall()
        clean_text = ' '.join([t.strip() for t in text_content if t.strip()])
        
        # Extract structured content
        headings = {
            'h1': response.css('h1::text').getall(),
            'h2': response.css('h2::text').getall(),
            'h3': response.css('h3::text').getall(),
        }
        
        # Extract images
        images = []
        for img in response.css('img'):
            src = img.css('::attr(src)').get()
            alt = img.css('::attr(alt)').get()
            if src:
                images.append({
                    'src': response.urljoin(src),
                    'alt': alt or '',
                })
        
        # Extract links
        links = []
        for link in response.css('a[href]'):
            href = link.css('::attr(href)').get()
            text = link.css('::text').get()
            if href:
                links.append({
                    'href': response.urljoin(href),
                    'text': text or '',
                })
        
        content.update({
            'text': clean_text,
            'word_count': len(clean_text.split()),
            'headings': headings,
            'images': images,
            'links': links,
            'html_length': len(response.text),
        })
        
        return content
    
    def detect_content_type(self, html: str, text: str) -> str:
        """Detect the type of content using pattern matching"""
        html_lower = html.lower()
        
        # Score each content type
        scores = {}
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, html_lower, re.IGNORECASE))
                score += matches
            scores[content_type] = score
        
        # Return the highest scoring type, or 'general' if no clear winner
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return 'general'
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text content"""
        try:
            if len(text.strip()) < 50:
                return 'unknown'
            
            # Use langdetect for language detection
            language = detect(text[:1000])  # Use first 1000 chars for detection
            return language
        except:
            return 'unknown'
    
    def calculate_content_score(self, content: Dict, metadata: Dict, content_type: str) -> float:
        """Calculate AI-based content quality score"""
        try:
            # Use the ML content scorer
            score = self.content_scorer.score_content(
                text=content.get('text', ''),
                title=metadata.get('title', ''),
                content_type=content_type,
                word_count=content.get('word_count', 0),
                image_count=len(content.get('images', [])),
                link_count=len(content.get('links', [])),
            )
            
            # Additional heuristic factors
            
            # Length bonus (sweet spot around 800-2000 words)
            word_count = content.get('word_count', 0)
            if 800 <= word_count <= 2000:
                score += 0.1
            elif word_count < 200:
                score -= 0.2
            
            # Structure bonus (good heading hierarchy)
            headings = content.get('headings', {})
            if headings.get('h1') and headings.get('h2'):
                score += 0.05
            
            # Metadata bonus
            if metadata.get('description'):
                score += 0.05
            if metadata.get('author'):
                score += 0.03
            
            # Content type specific adjustments
            if content_type == 'article':
                score += 0.1
            elif content_type == 'news':
                score += 0.05
            elif content_type == 'social':
                score -= 0.1  # Social content often less substantial
            
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating content score: {e}")
            return 0.5  # Default score
    
    def extract_links(self, response: Response) -> List[str]:
        """Extract all valid links from the page"""
        links = []
        
        for link in response.css('a[href]'):
            href = link.css('::attr(href)').get()
            if href:
                absolute_url = response.urljoin(href)
                
                # Basic URL validation
                if self.is_valid_url(absolute_url):
                    links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def prioritize_links(self, links: List[str], parent_score: float, parent_url: str) -> List[Tuple[str, float]]:
        """Prioritize links based on various factors"""
        prioritized = []
        
        for url in links:
            priority = self.calculate_link_priority(url, parent_score, parent_url)
            prioritized.append((url, priority))
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    def calculate_link_priority(self, url: str, parent_score: float, parent_url: str) -> float:
        """Calculate priority score for a link"""
        priority = 0.5  # Base priority
        
        # Parent page score influence
        priority += parent_score * 0.3
        
        # URL structure analysis
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Content type indicators in URL
        if any(keyword in path for keyword in ['article', 'post', 'news', 'blog']):
            priority += 0.2
        
        if any(keyword in path for keyword in ['about', 'contact', 'privacy', 'terms']):
            priority -= 0.3
        
        # Depth penalty (prefer shorter paths)
        path_depth = len([p for p in path.split('/') if p])
        priority -= path_depth * 0.05
        
        # Same domain bonus
        parent_domain = urlparse(parent_url).netloc
        if parsed_url.netloc == parent_domain:
            priority += 0.1
        
        # Popular TLD bonus
        if parsed_url.netloc.endswith(('.com', '.org', '.net', '.edu', '.gov')):
            priority += 0.05
        
        # Avoid certain file types
        if any(path.endswith(ext) for ext in ['.pdf', '.doc', '.zip', '.exe', '.dmg']):
            priority -= 0.4
        
        return max(0.0, min(1.0, priority))
    
    def should_follow_link(self, url: str, current_depth: int) -> bool:
        """Determine if a link should be followed"""
        # Max depth check
        if current_depth >= 5:  # Configurable max depth
            return False
        
        # Already crawled check
        if url in self.crawled_urls:
            return False
        
        # URL validation
        if not self.is_valid_url(url):
            return False
        
        # Domain filtering (if configured)
        if self.allowed_domains:
            domain = urlparse(url).netloc
            if not any(domain.endswith(allowed) for allowed in self.allowed_domains):
                return False
        
        return True
    
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme"""
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                not url.startswith('javascript:'),
                not url.startswith('mailto:'),
                not url.startswith('tel:'),
                '#' not in url,  # Skip anchors
            ])
        except:
            return False
    
    def update_domain_stats(self, url: str, score: float, content_type: str):
        """Update crawling statistics for the domain"""
        domain = urlparse(url).netloc
        
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                'pages_crawled': 0,
                'total_score': 0.0,
                'content_types': {},
                'first_seen': datetime.utcnow(),
            }
        
        stats = self.domain_stats[domain]
        stats['pages_crawled'] += 1
        stats['total_score'] += score
        stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
        stats['last_crawled'] = datetime.utcnow()
        stats['avg_score'] = stats['total_score'] / stats['pages_crawled']
    
    def closed(self, reason):
        """Called when spider closes"""
        logger.info(f"Spider closed: {reason}")
        logger.info(f"Crawled {len(self.crawled_urls)} URLs")
        logger.info(f"Domain stats: {self.domain_stats}")
        
        # Save crawl statistics to database
        asyncio.create_task(self.save_crawl_stats())
    
    async def save_crawl_stats(self):
        """Save crawling statistics to database"""
        try:
            stats = {
                'session_start': self.session_start,
                'session_end': datetime.utcnow(),
                'urls_crawled': len(self.crawled_urls),
                'domain_stats': self.domain_stats,
                'avg_content_score': np.mean(list(self.url_scores.values())) if self.url_scores else 0,
            }
            
            # Save to database (implementation depends on your database schema)
            await DatabaseManager.save_crawl_session(stats)
            
        except Exception as e:
            logger.error(f"Error saving crawl stats: {e}")


# Custom settings for different spider modes
class NewsSpider(AstraFindSpider):
    """Specialized spider for news content"""
    name = "astrafind_news"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prioritize news content
        self.content_patterns['news'] = [
            r'class="[^"]*news[^"]*"',
            r'class="[^"]*story[^"]*"',
            r'<time[^>]*>',
            r'datetime=',
            r'published',
            r'article',
        ]


class AcademicSpider(AstraFindSpider):
    """Specialized spider for academic content"""
    name = "astrafind_academic"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Focus on academic domains
        self.allowed_domains = [
            'arxiv.org',
            'scholar.google.com',
            'pubmed.ncbi.nlm.nih.gov',
            '.edu',
            '.ac.uk',
        ]