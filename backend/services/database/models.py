from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Any
import bcrypt
from ...utils.config import settings

Base = declarative_base()

class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    language_preference = Column(String(10), default='en')
    timezone = Column(String(50), default='UTC')
    
    # Privacy settings
    data_collection_consent = Column(Boolean, default=False)
    personalization_enabled = Column(Boolean, default=True)
    search_history_enabled = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    searches = relationship("SearchQuery", back_populates="user", cascade="all, delete-orphan")
    favorites = relationship("UserFavorite", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_created_at', 'created_at'),
    )
    
    def set_password(self, password: str):
        """Hash and set password"""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Check if password is correct"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': str(self.id),
            'email': self.email,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'language_preference': self.language_preference,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Document(Base):
    """Crawled and indexed document model"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    url_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hash of URL
    title = Column(Text)
    content = Column(Text)
    meta_description = Column(Text)
    language = Column(String(10), nullable=False, index=True)
    content_type = Column(String(50), default='text/html', index=True)
    
    # Content analysis
    word_count = Column(Integer, default=0)
    content_quality_score = Column(Float, default=0.0, index=True)
    readability_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    
    # SEO and metadata
    keywords = Column(ARRAY(String), default=[])
    entities = Column(JSON, default={})  # Named entities extracted
    topics = Column(ARRAY(String), default=[])
    
    # Crawling metadata
    crawled_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_modified = Column(DateTime)
    content_hash = Column(String(64), index=True)  # SHA-256 hash of content
    
    # Status and flags
    is_indexed = Column(Boolean, default=False, index=True)
    is_blocked = Column(Boolean, default=False, index=True)
    misinformation_score = Column(Float, default=0.0, index=True)
    bias_score = Column(Float, default=0.0)
    
    # Performance metrics
    page_rank_score = Column(Float, default=0.0, index=True)
    popularity_score = Column(Float, default=0.0, index=True)
    freshness_score = Column(Float, default=1.0, index=True)
    
    # Source information
    domain = Column(String(255), nullable=False, index=True)
    source_credibility = Column(Float, default=0.5, index=True)
    
    # Full-text search
    search_vector = Column(TSVECTOR)
    
    # Relationships
    outlinks = relationship("DocumentLink", foreign_keys="DocumentLink.source_document_id", back_populates="source_document")
    inlinks = relationship("DocumentLink", foreign_keys="DocumentLink.target_document_id", back_populates="target_document")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_documents_url_hash', 'url_hash'),
        Index('idx_documents_domain', 'domain'),
        Index('idx_documents_language', 'language'),
        Index('idx_documents_crawled_at', 'crawled_at'),
        Index('idx_documents_quality_score', 'content_quality_score'),
        Index('idx_documents_search_vector', 'search_vector', postgresql_using='gin'),
        Index('idx_documents_composite_quality', 'content_quality_score', 'page_rank_score', 'freshness_score'),
        CheckConstraint('content_quality_score >= 0 AND content_quality_score <= 1', name='valid_quality_score'),
        CheckConstraint('misinformation_score >= 0 AND misinformation_score <= 1', name='valid_misinformation_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': str(self.id),
            'url': self.url,
            'title': self.title,
            'meta_description': self.meta_description,
            'language': self.language,
            'content_type': self.content_type,
            'word_count': self.word_count,
            'content_quality_score': self.content_quality_score,
            'keywords': self.keywords,
            'topics': self.topics,
            'crawled_at': self.crawled_at.isoformat() if self.crawled_at else None,
            'domain': self.domain
        }

class DocumentEmbedding(Base):
    """Document embeddings for semantic search"""
    __tablename__ = "document_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    embedding_type = Column(String(50), nullable=False)  # 'title', 'content', 'combined'
    model_name = Column(String(100), nullable=False)
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    vector_dimension = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")
    
    __table_args__ = (
        Index('idx_embeddings_document_id', 'document_id'),
        Index('idx_embeddings_type', 'embedding_type'),
        UniqueConstraint('document_id', 'embedding_type', 'model_name', name='unique_document_embedding'),
    )

class DocumentLink(Base):
    """Links between documents for PageRank calculation"""
    __tablename__ = "document_links"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    target_document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    anchor_text = Column(Text)
    link_context = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_document_id], back_populates="outlinks")
    target_document = relationship("Document", foreign_keys=[target_document_id], back_populates="inlinks")
    
    __table_args__ = (
        Index('idx_links_source', 'source_document_id'),
        Index('idx_links_target', 'target_document_id'),
        UniqueConstraint('source_document_id', 'target_document_id', name='unique_document_link'),
    )

class SearchQuery(Base):
    """User search queries and analytics"""
    __tablename__ = "search_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    session_id = Column(String(255), nullable=False, index=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(20), default='text', nullable=False)  # 'text', 'voice', 'image'
    language = Column(String(10), default='en')
    search_context = Column(JSON)  # Additional context for the search
    
    # Results and performance
    results_count = Column(Integer, default=0)
    response_time_ms = Column(Integer)
    
    # User interaction
    clicked_results = Column(JSON, default={})  # Document IDs and click positions
    result_satisfaction = Column(Float)  # User feedback score
    
    # Search personalization
    user_location = Column(String(100))
    user_device = Column(String(50))
    user_agent = Column(Text)
    
    # Privacy and compliance
    anonymized = Column(Boolean, default=False)
    retention_until = Column(DateTime)  # When to delete for GDPR compliance
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="searches")
    
    __table_args__ = (
        Index('idx_search_queries_user_id', 'user_id'),
        Index('idx_search_queries_created_at', 'created_at'),
        Index('idx_search_queries_session', 'session_id'),
        Index('idx_search_queries_type', 'query_type'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': str(self.id),
            'query_text': self.query_text,
            'query_type': self.query_type,
            'language': self.language,
            'results_count': self.results_count,
            'response_time_ms': self.response_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserFavorite(Base):
    """User favorite documents"""
    __tablename__ = "user_favorites"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    
    # Optional user notes
    notes = Column(Text)
    tags = Column(ARRAY(String), default=[])
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="favorites")
    document = relationship("Document")
    
    __table_args__ = (
        Index('idx_favorites_user_id', 'user_id'),
        Index('idx_favorites_document_id', 'document_id'),
        UniqueConstraint('user_id', 'document_id', name='unique_user_favorite'),
    )

class UserPreference(Base):
    """User preferences and personalization settings"""
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Preference details
    preference_key = Column(String(100), nullable=False)
    preference_value = Column(JSON)
    preference_type = Column(String(50), default='user_setting')  # 'user_setting', 'ml_model', 'search_behavior'
    
    # Metadata
    source = Column(String(50), default='manual')  # 'manual', 'inferred', 'ml_model'
    confidence_score = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        Index('idx_preferences_user_id', 'user_id'),
        Index('idx_preferences_key', 'preference_key'),
        UniqueConstraint('user_id', 'preference_key', name='unique_user_preference'),
    )

class CrawlerJob(Base):
    """Crawler job queue and status"""
    __tablename__ = "crawler_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String(2048), nullable=False, index=True)
    priority = Column(Integer, default=5, index=True)  # 1-10, higher is more important
    status = Column(String(20), default='pending', nullable=False, index=True)  # pending, running, completed, failed
    
    # Job configuration
    crawl_depth = Column(Integer, default=1)
    respect_robots_txt = Column(Boolean, default=True)
    follow_links = Column(Boolean, default=True)
    
    # Execution details
    assigned_worker = Column(String(100))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Results
    documents_crawled = Column(Integer, default=0)
    links_discovered = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scheduled_for = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_crawler_jobs_status', 'status'),
        Index('idx_crawler_jobs_priority', 'priority'),
        Index('idx_crawler_jobs_scheduled', 'scheduled_for'),
    )

class SystemMetric(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # 'counter', 'gauge', 'histogram'
    metric_value = Column(Float, nullable=False)
    
    # Dimensions/tags
    dimensions = Column(JSON, default={})
    
    # Metadata
    source_component = Column(String(100), nullable=False)
    unit = Column(String(20))
    
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_metrics_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_metrics_component', 'source_component'),
    )

class AuditLog(Base):
    """Audit log for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # 'login', 'search', 'data_access', etc.
    event_description = Column(Text)
    resource_type = Column(String(50))  # 'document', 'user', 'search_query'
    resource_id = Column(String(255))
    
    # Request details
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    request_method = Column(String(10))
    request_path = Column(String(500))
    
    # Response details
    status_code = Column(Integer)
    response_size = Column(Integer)
    
    # Security flags
    is_suspicious = Column(Boolean, default=False, index=True)
    risk_score = Column(Float, default=0.0)
    
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User")
    
    __table_args__ = (
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_event_type', 'event_type'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
        Index('idx_audit_logs_suspicious', 'is_suspicious'),
    )

class DataRetentionPolicy(Base):
    """Data retention policies for GDPR compliance"""
    __tablename__ = "data_retention_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_type = Column(String(100), nullable=False, unique=True)  # 'search_queries', 'user_data', etc.
    retention_period_days = Column(Integer, nullable=False)
    auto_delete = Column(Boolean, default=True)
    
    # Legal basis
    legal_basis = Column(String(100))  # 'consent', 'legitimate_interest', 'legal_obligation'
    jurisdiction = Column(String(50), default='EU')
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_retention_policies_type', 'data_type'),
    )

# Database utility functions
def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, email: str, username: str, password: str, **kwargs) -> User:
    """Create new user"""
    user = User(email=email, username=username, **kwargs)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_document_by_url(db: Session, url: str) -> Optional[Document]:
    """Get document by URL"""
    import hashlib
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return db.query(Document).filter(Document.url_hash == url_hash).first()

def create_document(db: Session, **kwargs) -> Document:
    """Create new document"""
    import hashlib
    
    # Generate URL hash
    url = kwargs.get('url')
    if url:
        kwargs['url_hash'] = hashlib.sha256(url.encode()).hexdigest()
    
    # Extract domain
    if url:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        kwargs['domain'] = parsed.netloc
    
    document = Document(**kwargs)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document

def record_search_query(db: Session, user_id: Optional[str], query: str, **kwargs) -> SearchQuery:
    """Record a search query"""
    search_query = SearchQuery(
        user_id=user_id,
        query_text=query,
        **kwargs
    )
    
    # Set retention date based on policy
    if settings.GDPR_ENABLED:
        retention_days = getattr(settings, 'SEARCH_QUERY_RETENTION_DAYS', 90)
        search_query.retention_until = datetime.utcnow() + timedelta(days=retention_days)
    
    db.add(search_query)
    db.commit()
    db.refresh(search_query)
    return search_query

def log_audit_event(db: Session, event_type: str, user_id: Optional[str] = None, **kwargs) -> AuditLog:
    """Log an audit event"""
    audit_log = AuditLog(
        user_id=user_id,
        event_type=event_type,
        **kwargs
    )
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)
    return audit_log

def cleanup_expired_data(db: Session) -> int:
    """Clean up expired data based on retention policies"""
    cleanup_count = 0
    
    # Clean up expired search queries
    expired_queries = db.query(SearchQuery).filter(
        SearchQuery.retention_until < datetime.utcnow()
    ).all()
    
    for query in expired_queries:
        # Anonymize instead of delete if user hasn't opted out
        query.user_id = None
        query.anonymized = True
        query.user_location = None
        query.user_agent = None
        cleanup_count += 1
    
    db.commit()
    return cleanup_count