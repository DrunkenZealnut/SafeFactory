"""SQLAlchemy models for user authentication and social login."""

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# ---------------------------------------------------------------------------
# Token encryption helpers (Fernet with key derived from SECRET_KEY)
# ---------------------------------------------------------------------------

_fernet_instance = None


def _get_fernet():
    """Return a Fernet instance keyed from SECRET_KEY (lazy-init, cached)."""
    global _fernet_instance
    if _fernet_instance is None:
        secret = os.environ.get('SECRET_KEY', '')
        if not secret:
            logging.warning('SECRET_KEY not set — token encryption disabled')
            return None
        # Derive a 32-byte key via SHA-256 and base64-encode for Fernet
        key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        _fernet_instance = Fernet(key)
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
            'profile_image': self.profile_image,
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
