"""SQLAlchemy models for user authentication, social login, and community."""

import base64
import logging
import os
import threading
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from sqlalchemy.exc import IntegrityError
from urllib.parse import urlparse

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _safe_url(url):
    """Return URL only if scheme is http or https, else None."""
    if url and urlparse(url).scheme in ('http', 'https'):
        return url
    return None

# ---------------------------------------------------------------------------
# Token encryption helpers (Fernet with key derived from SECRET_KEY)
# ---------------------------------------------------------------------------

_fernet_instance = None
_fernet_initialized = False
_fernet_lock = threading.Lock()


def _get_fernet():
    """Return a Fernet instance keyed from SECRET_KEY (lazy-init, cached)."""
    global _fernet_instance, _fernet_initialized
    if not _fernet_initialized:
        with _fernet_lock:
            if not _fernet_initialized:
                secret = os.environ.get('SECRET_KEY', '')
                if not secret:
                    logging.warning('SECRET_KEY not set — token encryption disabled')
                    _fernet_initialized = True
                    return None
                # Derive key using PBKDF2 (fixed salt for deterministic derivation)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'safefactory_token_encryption',
                    iterations=100_000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
                _fernet_instance = Fernet(key)
                _fernet_initialized = True
    return _fernet_instance


def encrypt_token(value):
    """Encrypt a token string. Returns the ciphertext or the original value on failure."""
    if not value:
        return value
    f = _get_fernet()
    if f is None:
        return value
    return f.encrypt(value.encode()).decode()


def decrypt_token(value):
    """Decrypt a token string. Returns the plaintext or the original value on failure."""
    if not value:
        return value
    f = _get_fernet()
    if f is None:
        return value
    try:
        return f.decrypt(value.encode()).decode()
    except (InvalidToken, Exception):
        # Likely a plaintext token from before encryption was enabled
        return value


class User(db.Model, UserMixin):
    """User account model."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    profile_image = db.Column(db.String(500), nullable=True)
    role = db.Column(db.String(20), nullable=False, default='user')
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_login = db.Column(db.DateTime, nullable=True)

    social_accounts = db.relationship(
        'SocialAccount', back_populates='user', cascade='all, delete-orphan',
    )

    def to_dict(self):
        """Serialize user to dictionary."""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'profile_image': _safe_url(self.profile_image),
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'providers': [sa.provider for sa in self.social_accounts],
        }


class SocialAccount(db.Model):
    """Social login account linked to a user."""

    __tablename__ = 'social_accounts'
    __table_args__ = (
        db.UniqueConstraint('provider', 'provider_user_id', name='uq_provider_user'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    provider = db.Column(db.String(50), nullable=False)  # google, kakao
    provider_user_id = db.Column(db.String(255), nullable=False)
    _access_token = db.Column('access_token', db.Text, nullable=True)
    _refresh_token = db.Column('refresh_token', db.Text, nullable=True)
    token_expires_at = db.Column(db.DateTime, nullable=True)
    provider_data = db.Column(db.JSON, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    @property
    def access_token(self):
        return decrypt_token(self._access_token)

    @access_token.setter
    def access_token(self, value):
        self._access_token = encrypt_token(value)

    @property
    def refresh_token(self):
        return decrypt_token(self._refresh_token)

    @refresh_token.setter
    def refresh_token(self, value):
        self._refresh_token = encrypt_token(value)

    user = db.relationship('User', back_populates='social_accounts')


# ---------------------------------------------------------------------------
# Community models
# ---------------------------------------------------------------------------

class Category(db.Model):
    """Community board category."""

    __tablename__ = 'categories'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    slug = db.Column(db.String(50), unique=True, nullable=False, index=True)
    description = db.Column(db.String(200), nullable=True)
    color = db.Column(db.String(7), nullable=False, default='#00d4ff')
    icon = db.Column(db.String(10), nullable=True)
    sort_order = db.Column(db.Integer, nullable=False, default=0)
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    posts = db.relationship('Post', back_populates='category', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'color': self.color,
            'icon': self.icon,
            'sort_order': self.sort_order,
        }


class Post(db.Model):
    """Community board post."""

    __tablename__ = 'posts'
    __table_args__ = (
        db.Index('ix_posts_category_created', 'category_id', 'created_at'),
        db.Index('ix_posts_user', 'user_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    category_id = db.Column(
        db.Integer, db.ForeignKey('categories.id'), nullable=False,
    )
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    view_count = db.Column(db.Integer, nullable=False, default=0)
    is_pinned = db.Column(db.Boolean, nullable=False, default=False)
    is_deleted = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime, nullable=True,
        onupdate=lambda: datetime.now(timezone.utc),
    )

    category = db.relationship('Category', back_populates='posts')
    author = db.relationship('User', backref='posts')
    attachments = db.relationship(
        'PostAttachment', back_populates='post', cascade='all, delete-orphan',
    )
    comments = db.relationship(
        'Comment', back_populates='post', cascade='all, delete-orphan',
        lazy='dynamic',
    )
    likes = db.relationship(
        'PostLike', back_populates='post', cascade='all, delete-orphan',
        lazy='dynamic',
    )

    @property
    def like_count(self):
        return self.likes.count()

    @property
    def comment_count(self):
        return self.comments.filter_by(is_deleted=False).count()

    def to_dict(self, include_content=False, _like_count=None, _comment_count=None):
        d = {
            'id': self.id,
            'category': self.category.to_dict() if self.category else None,
            'author': {
                'id': self.author.id,
                'name': self.author.name,
                'profile_image': _safe_url(self.author.profile_image),
            } if self.author else None,
            'title': self.title,
            'view_count': self.view_count,
            'like_count': _like_count if _like_count is not None else self.like_count,
            'comment_count': _comment_count if _comment_count is not None else self.comment_count,
            'is_pinned': self.is_pinned,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_content:
            d['content'] = self.content
            d['attachments'] = [a.to_dict() for a in self.attachments]
        return d


class PostAttachment(db.Model):
    """File attachment for a post."""

    __tablename__ = 'post_attachments'

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.Integer, db.ForeignKey('posts.id', ondelete='CASCADE'), nullable=False,
    )
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    mime_type = db.Column(db.String(100), nullable=False)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    post = db.relationship('Post', back_populates='attachments')

    def to_dict(self):
        from flask import request as _req
        path = f'/static/uploads/community/{self.filename}'
        try:
            base = _req.host_url.rstrip('/')
            url = f'{base}{path}'
        except RuntimeError:
            url = path
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'url': url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Comment(db.Model):
    """Comment on a post, supports nesting via parent_id."""

    __tablename__ = 'comments'
    __table_args__ = (
        db.Index('ix_comments_post', 'post_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.Integer, db.ForeignKey('posts.id', ondelete='CASCADE'), nullable=False,
    )
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    parent_id = db.Column(
        db.Integer, db.ForeignKey('comments.id', ondelete='CASCADE'), nullable=True,
    )
    content = db.Column(db.Text, nullable=False)
    is_deleted = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime, nullable=True,
        onupdate=lambda: datetime.now(timezone.utc),
    )

    post = db.relationship('Post', back_populates='comments')
    author = db.relationship('User', backref='comments')
    replies = db.relationship(
        'Comment', backref=db.backref('parent', remote_side='Comment.id'),
        lazy='dynamic',
    )

    MAX_REPLY_DEPTH = 5

    def to_dict(self, _depth=0, include_replies=False):
        d = {
            'id': self.id,
            'post_id': self.post_id,
            'author': {
                'id': self.author.id,
                'name': self.author.name,
                'profile_image': _safe_url(self.author.profile_image),
            } if self.author else None,
            'parent_id': self.parent_id,
            'content': self.content if not self.is_deleted else '삭제된 댓글입니다.',
            'is_deleted': self.is_deleted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'replies': [],
        }
        if include_replies and _depth < self.MAX_REPLY_DEPTH:
            d['replies'] = [
                r.to_dict(_depth=_depth + 1, include_replies=include_replies)
                for r in self.replies.filter_by(is_deleted=False).all()
            ]
        return d


class PostLike(db.Model):
    """Like on a post (unique per user+post)."""

    __tablename__ = 'post_likes'
    __table_args__ = (
        db.UniqueConstraint('post_id', 'user_id', name='uq_post_user_like'),
    )

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(
        db.Integer, db.ForeignKey('posts.id', ondelete='CASCADE'), nullable=False,
    )
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    post = db.relationship('Post', back_populates='likes')
    user = db.relationship('User')


def seed_categories():
    """Insert default community categories if table is empty."""
    if Category.query.first() is not None:
        return
    defaults = [
        Category(name='공지', slug='notice', description='관리자 공지사항',
                 color='#ff9800', icon='📢', sort_order=0),
        Category(name='질문', slug='qna', description='질문과 답변',
                 color='#00d4ff', icon='❓', sort_order=1),
        Category(name='정보공유', slug='info', description='유용한 정보 공유',
                 color='#4caf50', icon='💡', sort_order=2),
        Category(name='자유', slug='free', description='자유로운 대화',
                 color='#9c27b0', icon='💬', sort_order=3),
    ]
    try:
        db.session.add_all(defaults)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.warning("Failed to seed categories: %s", e)


# ---------------------------------------------------------------------------
# Admin models
# ---------------------------------------------------------------------------

class Document(db.Model):
    """Learning material catalog entry (synced with Pinecone vectors)."""

    __tablename__ = 'documents'
    __table_args__ = (
        db.UniqueConstraint('namespace', 'source_file', name='uq_ns_source'),
        db.Index('ix_documents_namespace', 'namespace'),
        db.Index('ix_documents_status', 'status'),
    )

    id = db.Column(db.Integer, primary_key=True)
    namespace = db.Column(db.String(100), nullable=False, default='')
    source_file = db.Column(db.String(500), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)
    vector_count = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(20), nullable=False, default='indexed')
    uploaded_by = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True,
    )
    metadata_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime, nullable=True,
        onupdate=lambda: datetime.now(timezone.utc),
    )

    uploader = db.relationship('User', foreign_keys=[uploaded_by])

    def to_dict(self):
        return {
            'id': self.id,
            'namespace': self.namespace,
            'source_file': self.source_file,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'vector_count': self.vector_count,
            'status': self.status,
            'uploaded_by': {
                'id': self.uploader.id,
                'name': self.uploader.name,
            } if self.uploader else None,
            'metadata': self.metadata_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AdminLog(db.Model):
    """Audit log for admin actions."""

    __tablename__ = 'admin_logs'
    __table_args__ = (
        db.Index('ix_admin_logs_created', 'created_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True,
    )
    action = db.Column(db.String(50), nullable=False)
    target_type = db.Column(db.String(30), nullable=True)
    target_id = db.Column(db.Integer, nullable=True)
    details = db.Column(db.JSON, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    admin = db.relationship('User', foreign_keys=[admin_id])

    def to_dict(self):
        return {
            'id': self.id,
            'admin': {
                'id': self.admin.id,
                'name': self.admin.name,
            } if self.admin else None,
            'action': self.action,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# System settings
# ---------------------------------------------------------------------------

class SystemSetting(db.Model):
    """Key-value store for system configuration (admin-managed)."""

    __tablename__ = 'system_settings'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False, index=True)
    value = db.Column(db.String(500), nullable=False)
    description = db.Column(db.String(200), nullable=True)
    category = db.Column(db.String(50), nullable=False, default='general')
    updated_at = db.Column(db.DateTime, nullable=True)
    updated_by = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True,
    )

    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'category': self.category,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


def seed_system_settings():
    """Insert default system settings if table is empty."""
    if SystemSetting.query.first() is not None:
        return
    defaults = [
        SystemSetting(
            key='llm_answer_model', value='gpt-4o-mini',
            description='답변 생성 모델', category='llm',
        ),
        SystemSetting(
            key='llm_answer_provider', value='openai',
            description='답변 생성 LLM 제공자', category='llm',
        ),
        SystemSetting(
            key='llm_answer_temperature', value='0.3',
            description='답변 생성 온도 (0.0~1.0)', category='llm',
        ),
        SystemSetting(
            key='llm_query_model', value='gpt-4o-mini',
            description='쿼리 강화 모델', category='llm',
        ),
        SystemSetting(
            key='llm_query_provider', value='openai',
            description='쿼리 강화 LLM 제공자', category='llm',
        ),
        SystemSetting(
            key='llm_context_model', value='gpt-4o-mini',
            description='컨텍스트 최적화 모델', category='llm',
        ),
        SystemSetting(
            key='llm_context_provider', value='openai',
            description='컨텍스트 최적화 LLM 제공자', category='llm',
        ),
        SystemSetting(
            key='embedding_model', value='text-embedding-3-small',
            description='임베딩 모델', category='embedding',
        ),
        SystemSetting(
            key='reranker_type', value='pinecone',
            description='리랭커 유형 (pinecone/local/lightweight)', category='reranker',
        ),
    ]
    try:
        db.session.add_all(defaults)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.warning("Failed to seed system settings: %s", e)


def ensure_provider_settings():
    """Add provider settings if they don't exist yet (safe for existing DBs)."""
    new_keys = [
        ('llm_answer_provider', 'openai', '답변 생성 LLM 제공자', 'llm'),
        ('llm_query_provider', 'openai', '쿼리 강화 LLM 제공자', 'llm'),
        ('llm_context_provider', 'openai', '컨텍스트 최적화 LLM 제공자', 'llm'),
    ]
    for key, value, desc, category in new_keys:
        if not SystemSetting.query.filter_by(key=key).first():
            try:
                db.session.add(SystemSetting(
                    key=key, value=value, description=desc, category=category,
                ))
                db.session.commit()
            except IntegrityError:
                db.session.rollback()  # another worker already inserted
            except Exception as e:
                db.session.rollback()
                logging.warning("Failed to ensure provider setting %s: %s", key, e)


def ensure_calculator_settings():
    """Add calculator rate settings if they don't exist yet (safe for existing DBs)."""
    new_keys = [
        # 국민연금
        ('calc.np_rate', '0.0475', '국민연금 근로자 요율', 'calculator'),
        ('calc.np_max_income', '6370000', '국민연금 기준소득월액 상한', 'calculator'),
        ('calc.np_min_income', '400000', '국민연금 기준소득월액 하한', 'calculator'),
        # 건강보험
        ('calc.hi_rate', '0.03595', '건강보험 근로자 요율', 'calculator'),
        ('calc.hi_max_income', '127725730', '건강보험 보수월액 상한', 'calculator'),
        ('calc.hi_min_income', '280528', '건강보험 보수월액 하한', 'calculator'),
        ('calc.hi_max_premium', '9183460', '건강보험료 월 상한', 'calculator'),
        ('calc.hi_min_premium', '20160', '건강보험료 월 하한', 'calculator'),
        # 장기요양보험
        ('calc.ltc_rate', '0.1314', '장기요양보험 요율 (건강보험료 대비)', 'calculator'),
        # 고용보험
        ('calc.ei_employee', '0.009', '고용보험 근로자 요율', 'calculator'),
        ('calc.ei_employer_base', '0.009', '고용보험 사업주 기본 요율', 'calculator'),
        ('calc.ei_under_150', '0.0025', '고용안정 150인 미만', 'calculator'),
        ('calc.ei_priority', '0.0045', '고용안정 우선지원대상', 'calculator'),
        ('calc.ei_150_to_999', '0.0065', '고용안정 150~999인', 'calculator'),
        ('calc.ei_over_1000', '0.0085', '고용안정 1000인 이상', 'calculator'),
        # 산재보험 부가금
        ('calc.ia_commute', '0.006', '출퇴근재해 요율', 'calculator'),
        ('calc.ia_wage_claim', '0.0006', '임금채권부담금 요율', 'calculator'),
        ('calc.ia_asbestos', '0.0003', '석면피해구제분담금 요율', 'calculator'),
        # 최저임금
        ('calc.min_wage_year', '2026', '현행 최저임금 적용 연도', 'calculator'),
        ('calc.min_wage_2026', '10320', '2026년 최저시급 (원)', 'calculator'),
        ('calc.min_wage_2025', '10030', '2025년 최저시급 (원)', 'calculator'),
        # 메타
        ('calc.rates_updated_at', '2026-01-01', '요율 마지막 확인일', 'calculator'),
        ('calc.rates_year', '2026', '적용 요율 기준년도', 'calculator'),
    ]
    for key, value, desc, category in new_keys:
        if not SystemSetting.query.filter_by(key=key).first():
            try:
                db.session.add(SystemSetting(
                    key=key, value=value, description=desc, category=category,
                ))
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
            except Exception as e:
                db.session.rollback()
                logging.warning("Failed to ensure calculator setting %s: %s", key, e)


# ---------------------------------------------------------------------------
# News models
# ---------------------------------------------------------------------------

class NewsArticle(db.Model):
    """Labor safety news article (admin-curated)."""

    __tablename__ = 'news_articles'
    __table_args__ = (
        db.Index('ix_news_published', 'published_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    summary = db.Column(db.String(500), nullable=True)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    source_name = db.Column(db.String(100), nullable=True)
    source_url = db.Column(db.String(500), nullable=True)
    source_image = db.Column(db.String(2000), nullable=True)
    view_count = db.Column(db.Integer, nullable=False, default=0)
    is_published = db.Column(db.Boolean, nullable=False, default=True)
    author_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True,
    )
    published_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime, nullable=True,
        onupdate=lambda: datetime.now(timezone.utc),
    )

    author = db.relationship('User', foreign_keys=[author_id])

    CATEGORIES = {
        'accident': '산재사고',
        'regulation': '법령개정',
        'policy': '정책동향',
        'technology': '안전기술',
        'general': '일반',
    }

    def to_dict(self, include_content=False):
        d = {
            'id': self.id,
            'title': self.title,
            'summary': self.summary,
            'category': self.category,
            'category_label': self.CATEGORIES.get(self.category, self.category),
            'source_name': self.source_name,
            'source_url': _safe_url(self.source_url),
            'source_image': _safe_url(self.source_image),
            'view_count': self.view_count,
            'is_published': self.is_published,
            'author': {
                'id': self.author.id,
                'name': self.author.name,
            } if self.author else None,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        if include_content:
            d['content'] = self.content
        return d
