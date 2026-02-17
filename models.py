"""SQLAlchemy models for user authentication, social login, and community."""

import base64
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from urllib.parse import urlparse

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _safe_image_url(url):
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
                # Derive a 32-byte key via SHA-256 and base64-encode for Fernet
                key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
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
            'profile_image': _safe_image_url(self.profile_image),
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
    provider_data = db.Column(db.JSON, nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

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

    def to_dict(self, include_content=False):
        d = {
            'id': self.id,
            'category': self.category.to_dict() if self.category else None,
            'author': {
                'id': self.author.id,
                'name': self.author.name,
                'profile_image': _safe_image_url(self.author.profile_image),
            } if self.author else None,
            'title': self.title,
            'view_count': self.view_count,
            'like_count': self.like_count,
            'comment_count': self.comment_count,
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

    def to_dict(self, _depth=0):
        d = {
            'id': self.id,
            'post_id': self.post_id,
            'author': {
                'id': self.author.id,
                'name': self.author.name,
                'profile_image': _safe_image_url(self.author.profile_image),
            } if self.author else None,
            'parent_id': self.parent_id,
            'content': self.content if not self.is_deleted else '삭제된 댓글입니다.',
            'is_deleted': self.is_deleted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if _depth < self.MAX_REPLY_DEPTH:
            d['replies'] = [
                r.to_dict(_depth=_depth + 1)
                for r in self.replies.filter_by(is_deleted=False).all()
            ]
        else:
            d['replies'] = []
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
