"""Labor safety news API endpoints."""

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import requests as http_requests
from flask import request
from flask_login import current_user

from api.response import error_response, escape_like, success_response
from api.v1 import v1_bp
from api.v1.admin import admin_required
from models import NewsArticle, db


# ---------------------------------------------------------------------------
# Link preview helpers
# ---------------------------------------------------------------------------

_OG_RE = re.compile(
    r'<meta\s+[^>]*(?:property|name)\s*=\s*["\']og:(\w+)["\'][^>]*content\s*=\s*["\']([^"\']*)["\']'
    r'|<meta\s+[^>]*content\s*=\s*["\']([^"\']*)["\'][^>]*(?:property|name)\s*=\s*["\']og:(\w+)["\']',
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
_DESC_RE = re.compile(
    r'<meta\s+[^>]*name\s*=\s*["\']description["\'][^>]*content\s*=\s*["\']([^"\']*)["\']'
    r'|<meta\s+[^>]*content\s*=\s*["\']([^"\']*)["\'][^>]*name\s*=\s*["\']description["\']',
    re.IGNORECASE,
)
_IMG_RE = re.compile(r'<img\s+[^>]*src\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def _resolve_safe_ip(hostname):
    """Resolve hostname and return first public IP, or None if any address is private.

    All resolved addresses are checked — if even one is private/loopback/reserved
    the hostname is rejected.  The returned IP can be used directly in HTTP requests
    to prevent DNS rebinding (where a second resolution returns a different address).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
        if not infos:
            return None
        for info in infos:
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return None
        return infos[0][4][0]
    except (socket.gaierror, ValueError):
        return None


def _make_ip_url(original_url, resolved_ip):
    """Replace hostname with resolved IP in the URL to pin the connection.

    The original Host header must be set separately so the target server
    can route the request correctly.
    """
    parsed = urlparse(original_url)
    # For IPv6 addresses, wrap in brackets
    ip_host = f'[{resolved_ip}]' if ':' in resolved_ip else resolved_ip
    if parsed.port:
        netloc = f'{ip_host}:{parsed.port}'
    else:
        netloc = ip_host
    return parsed._replace(netloc=netloc).geturl()


def _safe_get(url, headers, timeout):
    """Resolve hostname, validate IP, and GET using the pinned IP.

    Returns the Response object or None if the URL targets a private IP.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return None
    resolved_ip = _resolve_safe_ip(parsed.hostname or '')
    if not resolved_ip:
        return None
    pinned_url = _make_ip_url(url, resolved_ip)
    req_headers = {**headers, 'Host': parsed.hostname}
    return http_requests.get(
        pinned_url, headers=req_headers, timeout=timeout,
        allow_redirects=False, stream=True,
        verify=True,
    )


def _fetch_og_metadata(url, timeout=5):
    """Fetch a URL and extract Open Graph metadata."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; SafeFactory/1.0; +link-preview)',
        'Accept': 'text/html',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.5',
    }
    try:
        resp = _safe_get(url, headers, timeout)
        if resp is None:
            return None
        # Handle redirects manually with IP validation on every hop
        redirect_count = 0
        while resp.status_code in (301, 302, 303, 307, 308) and redirect_count < 5:
            redirect_url = resp.headers.get('Location')
            if not redirect_url:
                return None
            resp = _safe_get(redirect_url, headers, timeout)
            if resp is None:
                return None
            redirect_count += 1
        resp.raise_for_status()

        content_type = resp.headers.get('content-type', '')
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            return None

        # Read at most 200KB using streaming to avoid loading huge responses
        chunks = []
        bytes_read = 0
        for chunk in resp.iter_content(chunk_size=8192):
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read >= 200_000:
                break
        resp.close()
        raw = b''.join(chunks)[:200_000]
        html = raw.decode(resp.apparent_encoding or 'utf-8', errors='replace')
    except Exception:
        return None

    # Extract OG tags
    og = {}
    for m in _OG_RE.finditer(html):
        key = m.group(1) or m.group(4)
        val = m.group(2) or m.group(3)
        if key and val:
            og[key.lower()] = val.strip()

    title = og.get('title', '')
    description = og.get('description', '')
    image = og.get('image', '')
    site_name = og.get('site_name', '')

    # Fallback to <title> and <meta name="description">
    if not title:
        m = _TITLE_RE.search(html)
        if m:
            title = re.sub(r'<[^>]+>', '', m.group(1)).strip()

    if not description:
        m = _DESC_RE.search(html)
        if m:
            description = (m.group(1) or m.group(2) or '').strip()

    # Fallback: first <img> with reasonable src
    if not image:
        m = _IMG_RE.search(html)
        if m:
            src = m.group(1)
            if src.startswith(('http://', 'https://')):
                image = src

    # Make relative image URLs absolute
    if image and not image.startswith(('http://', 'https://')):
        base = f'{parsed.scheme}://{parsed.netloc}'
        image = base + ('/' if not image.startswith('/') else '') + image

    if not title and not description:
        return None

    if not site_name:
        site_name = parsed.hostname or ''

    return {
        'url': url,
        'title': title[:300],
        'description': description[:500],
        'image': image[:2000] if image else None,
        'site_name': site_name[:100],
    }


@v1_bp.route('/news/link-preview', methods=['POST'])
@admin_required
def api_news_link_preview():
    """Fetch Open Graph metadata for a URL (admin only)."""
    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    url = (data.get('url') or '').strip()
    if not url or not url.startswith(('http://', 'https://')):
        return error_response('유효한 URL이 아닙니다.', 400)
    if len(url) > 2000:
        return error_response('URL이 너무 깁니다.', 400)

    meta = _fetch_og_metadata(url)
    if not meta:
        return error_response('링크 정보를 가져올 수 없습니다.', 422)

    return success_response(data=meta)


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@v1_bp.route('/news/categories', methods=['GET'])
def api_news_categories():
    """List news categories."""
    cats = [{'slug': k, 'label': v} for k, v in NewsArticle.CATEGORIES.items()]
    return success_response(data=cats)


@v1_bp.route('/news/articles', methods=['GET'])
def api_news_articles():
    """List published news articles with pagination, category filter, and search."""
    try:
        page = min(10000, max(1, request.args.get('page', 1, type=int)))
        per_page = min(50, max(1, request.args.get('per_page', 20, type=int)))
        category = request.args.get('category', '').strip()
        search_query = request.args.get('search', '').strip()

        query = NewsArticle.query.filter_by(is_published=True)

        if category and category in NewsArticle.CATEGORIES:
            query = query.filter_by(category=category)

        if search_query:
            like_pattern = f'%{escape_like(search_query)}%'
            query = query.filter(
                db.or_(
                    NewsArticle.title.ilike(like_pattern, escape='\\'),
                    NewsArticle.summary.ilike(like_pattern, escape='\\'),
                    NewsArticle.content.ilike(like_pattern, escape='\\'),
                )
            )

        query = query.order_by(NewsArticle.published_at.desc())
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return success_response(data={
            'articles': [a.to_dict() for a in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
        })
    except Exception:
        logging.exception('News list failed')
        return error_response('뉴스 목록 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/news/articles/<int:article_id>', methods=['GET'])
def api_news_article_detail(article_id):
    """Get news article detail. Increments view count.

    Admins can view unpublished articles; regular users only see published ones.
    """
    query = NewsArticle.query.filter_by(id=article_id)
    is_admin = current_user.is_authenticated and current_user.role == 'admin'
    if not is_admin:
        query = query.filter_by(is_published=True)
    article = query.first()
    if not article:
        return error_response('뉴스를 찾을 수 없습니다.', 404)

    # Atomic view-count increment (non-critical, suppress errors)
    try:
        NewsArticle.query.filter_by(id=article_id).update(
            {NewsArticle.view_count: NewsArticle.view_count + 1}
        )
        db.session.commit()
        article.view_count = (article.view_count or 0) + 1
    except Exception:
        db.session.rollback()

    return success_response(data=article.to_dict(include_content=True))


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@v1_bp.route('/news/articles', methods=['POST'])
@admin_required
def api_news_create_article():
    """Create a news article (admin only)."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        title = (data.get('title') or '').strip()
        content = (data.get('content') or '').strip()
        category = (data.get('category') or 'general').strip()

        if not title or len(title) < 2:
            return error_response('제목은 2자 이상이어야 합니다.', 400)
        if len(title) > 300:
            return error_response('제목은 300자 이하여야 합니다.', 400)
        if not content:
            return error_response('내용을 입력해주세요.', 400)
        if len(content) > 50000:
            return error_response('내용은 50,000자 이하여야 합니다.', 400)
        if category not in NewsArticle.CATEGORIES:
            return error_response('유효하지 않은 카테고리입니다.', 400)

        summary = (data.get('summary') or '').strip() or None
        source_name = (data.get('source_name') or '').strip() or None
        source_url = (data.get('source_url') or '').strip() or None
        source_image = (data.get('source_image') or '').strip() or None

        if summary and len(summary) > 500:
            return error_response('요약은 500자 이하여야 합니다.', 400)
        if source_name and len(source_name) > 200:
            return error_response('출처명은 200자 이하여야 합니다.', 400)
        if source_url and len(source_url) > 2000:
            return error_response('출처 URL은 2,000자 이하여야 합니다.', 400)
        if source_image and len(source_image) > 2000:
            return error_response('이미지 URL은 2,000자 이하여야 합니다.', 400)

        article = NewsArticle(
            title=title,
            content=content,
            summary=summary,
            category=category,
            source_name=source_name,
            source_url=source_url,
            source_image=source_image,
            is_published=data.get('is_published', True),
            author_id=current_user.id,
        )
        db.session.add(article)
        db.session.commit()

        return success_response(
            data=article.to_dict(include_content=True),
            message='뉴스가 등록되었습니다.',
        )
    except Exception:
        db.session.rollback()
        logging.exception('News creation failed')
        return error_response('뉴스 등록 중 오류가 발생했습니다.', 500)


@v1_bp.route('/news/articles/<int:article_id>', methods=['PUT'])
@admin_required
def api_news_update_article(article_id):
    """Update a news article (admin only)."""
    try:
        article = db.session.get(NewsArticle, article_id)
        if not article:
            return error_response('뉴스를 찾을 수 없습니다.', 404)

        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        # Validate all inputs before mutating the ORM object
        updates = {}
        if 'title' in data:
            title = (data['title'] or '').strip()
            if len(title) < 2 or len(title) > 300:
                return error_response('제목은 2~300자여야 합니다.', 400)
            updates['title'] = title
        if 'content' in data:
            content = (data['content'] or '').strip()
            if not content:
                return error_response('내용을 입력해주세요.', 400)
            if len(content) > 50000:
                return error_response('내용은 50,000자 이하여야 합니다.', 400)
            updates['content'] = content
        if 'summary' in data:
            summary = (data['summary'] or '').strip() or None
            if summary and len(summary) > 500:
                return error_response('요약은 500자 이하여야 합니다.', 400)
            updates['summary'] = summary
        if 'category' in data:
            if data['category'] not in NewsArticle.CATEGORIES:
                return error_response('유효하지 않은 카테고리입니다.', 400)
            updates['category'] = data['category']
        if 'source_name' in data:
            source_name = (data['source_name'] or '').strip() or None
            if source_name and len(source_name) > 200:
                return error_response('출처명은 200자 이하여야 합니다.', 400)
            updates['source_name'] = source_name
        if 'source_url' in data:
            source_url = (data['source_url'] or '').strip() or None
            if source_url and len(source_url) > 2000:
                return error_response('출처 URL은 2,000자 이하여야 합니다.', 400)
            updates['source_url'] = source_url
        if 'source_image' in data:
            source_image = (data['source_image'] or '').strip() or None
            if source_image and len(source_image) > 2000:
                return error_response('이미지 URL은 2,000자 이하여야 합니다.', 400)
            updates['source_image'] = source_image
        if 'is_published' in data:
            updates['is_published'] = bool(data['is_published'])

        # Apply validated updates
        for field, value in updates.items():
            setattr(article, field, value)

        db.session.commit()
        return success_response(
            data=article.to_dict(include_content=True),
            message='뉴스가 수정되었습니다.',
        )
    except Exception:
        db.session.rollback()
        logging.exception('News update failed')
        return error_response('뉴스 수정 중 오류가 발생했습니다.', 500)


@v1_bp.route('/news/articles/<int:article_id>', methods=['DELETE'])
@admin_required
def api_news_delete_article(article_id):
    """Delete a news article (admin only)."""
    try:
        article = db.session.get(NewsArticle, article_id)
        if not article:
            return error_response('뉴스를 찾을 수 없습니다.', 404)

        db.session.delete(article)
        db.session.commit()
        return success_response(message='뉴스가 삭제되었습니다.')
    except Exception:
        db.session.rollback()
        logging.exception('News deletion failed')
        return error_response('뉴스 삭제 중 오류가 발생했습니다.', 500)
