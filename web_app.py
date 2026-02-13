#!/usr/bin/env python3
"""
Pinecone Agent Web Interface
Flask-based web UI for Pinecone vector database operations with RAG support.
"""

# SSL Certificate configuration (must be done before importing httpx/urllib3)
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import re
import json
import logging
import unicodedata
import httpx
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI

# Import calculator modules
from calculator import WageCalculator, InsuranceCalculator, CompanySize, IndustryType

# Import RAG enhancement modules
from src.query_enhancer import QueryEnhancer
from src.context_optimizer import ContextOptimizer
from src.reranker import get_reranker
from src.hybrid_searcher import get_hybrid_searcher
from msds_client import MsdsApiClient

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Documents folder path for serving images
DOCUMENTS_PATH = Path(__file__).parent / "documents"

# Domain-specific system prompts
DOMAIN_PROMPTS = {
    "laborlaw": """당신은 한국 노동법 전문가입니다.
제공된 법률 문서들을 바탕으로 사용자의 질문에 대해 정확한 답변을 제공합니다.

## 계산기 도구 사용 (중요)

**임금 및 보험료 계산이 필요한 경우 반드시 제공된 함수를 사용하세요:**

1. **calculate_wage**: 실수령액, 4대보험료, 소득세 계산
   - 사용 시기: 연봉/월급에서 실제 받는 급여를 계산할 때
   - 예시: "연봉 5000만원 실수령액은?", "월급 300만원 세금 얼마?"

2. **calculate_insurance**: 4대보험료 상세 계산
   - 사용 시기: 업종별/규모별 보험료를 자세히 알고 싶을 때
   - 예시: "건설업 4대보험료는?", "300만원 국민연금 얼마?"

**함수를 사용한 후에는 결과를 마크다운 표로 정리하여 보기 좋게 제시하세요.**

## 필수 계산 규칙 (수동 계산 시)

### 주휴수당
- 조건: 주 15시간 이상 근무 시 의무 지급
- 공식: 주휴수당 = (주당 근무시간 ÷ 40) × 8 × 시급
- **주급/월급 계산 시 반드시 주휴수당을 포함하여 답변**

### 월급 계산
- 월 환산: 주당시간 × 4.345주
- 주휴 포함 월급: (주당 근무시간 + 주휴시간) × 4.345 × 시급

### 가산수당
- 연장근로(8시간 초과): 통상임금 × 1.5배
- 야간근로(22시~06시): 통상임금 × 1.5배
- 휴일근로: 통상임금 × 1.5배 (8시간 초과 시 2.0배)

## 답변 지침
1. **임금/보험료 계산 시 함수를 우선 사용**하고, 결과를 표로 정리
2. 계산 과정을 단계별로 설명
3. 관련 법조항 인용 시 [1], [2] 형식 사용
4. 마크다운 표로 계산 내역 정리
5. 문서에 없는 내용은 추측하지 마세요""",
}

# Default system prompt (semiconductor/general domain)
DEFAULT_SYSTEM_PROMPT = """당신은 반도체 기술 전문가입니다.
제공된 문서들을 바탕으로 사용자의 질문에 대해 종합적이고 정확한 답변을 제공합니다.

답변 작성 지침:
1. 제공된 문서 내용을 기반으로 답변하세요
2. **중요**: 각 정보의 출처를 반드시 인용 번호로 표시하세요. 예: "CVD 공정은 화학 기상 증착 방식입니다[1]."
3. 인용 형식: 문장 끝에 [1], [2] 등의 번호를 붙여 어떤 문서에서 가져온 정보인지 명시하세요
4. 여러 문서의 내용을 종합할 때는 [1][3]처럼 복수 인용도 가능합니다
5. 기술 용어는 한글과 영문을 병기하세요
6. 핵심 포인트를 명확하게 구분하세요
7. 문서에 없는 내용은 추측하지 마세요
8. 마크다운 형식으로 가독성 있게 작성하세요
9. **중요**: 이미지는 절대 직접 삽입하지 마세요 (![image](...) 형식 사용 금지). 관련 이미지는 시스템이 자동으로 표시합니다."""

# Global instances (lazy initialization)
_agent = None
_openai_client = None
_query_enhancer = None
_context_optimizer = None
_reranker = None
_hybrid_searcher = None

def get_openai_client():
    """Get or create OpenAI client with SSL certificate configuration."""
    global _openai_client
    if _openai_client is None:
        # Create httpx client with explicit SSL certificate verification
        http_client = httpx.Client(verify=certifi.where())
        _openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=http_client
        )
    return _openai_client

def get_agent():
    """Get or create the PineconeAgent instance."""
    global _agent
    if _agent is None:
        from src.agent import PineconeAgent
        _agent = PineconeAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
            create_index_if_not_exists=False
        )
    return _agent


def get_query_enhancer():
    """Get or create QueryEnhancer instance."""
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = QueryEnhancer(os.getenv("OPENAI_API_KEY"))
    return _query_enhancer


def get_context_optimizer():
    """Get or create ContextOptimizer instance."""
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextOptimizer(os.getenv("OPENAI_API_KEY"))
    return _context_optimizer


def get_pinecone_client():
    """Get Pinecone client from agent or create standalone."""
    try:
        agent = get_agent()
        return agent.pinecone_uploader.pc
    except Exception:
        try:
            from pinecone import Pinecone
            return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        except Exception:
            return None


def get_reranker_instance():
    """Get or create Reranker instance (prefers Pinecone Inference API)."""
    global _reranker
    if _reranker is None:
        pc = get_pinecone_client()
        _reranker = get_reranker(use_cross_encoder=True, pinecone_client=pc)
    return _reranker


def get_hybrid_searcher_instance():
    """Get or create HybridSearcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = get_hybrid_searcher()
    return _hybrid_searcher

def get_uploader():
    """Get PineconeUploader for stats."""
    from src.pinecone_uploader import PineconeUploader
    return PineconeUploader(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
        create_if_not_exists=False
    )


# ============================================
# Calculator Functions for GPT Function Calling
# ============================================

def calculate_wage(
    salary_type: str,
    amount: int,
    tax_free_monthly: int = 0,
    dependents: int = 1,
    children_8_to_20: int = 0,
    company_size: str = 'small'
) -> dict:
    """
    임금 계산 (실수령액, 4대보험료, 세금)

    Args:
        salary_type: '연봉' 또는 '월급'
        amount: 급여액 (원)
        tax_free_monthly: 월 비과세액 (최대 200,000원)
        dependents: 부양가족 수 (본인 포함, 기본 1명)
        children_8_to_20: 8~20세 자녀 수 (기본 0명)
        company_size: 회사 규모 ('small': 150인 미만, 'medium': 150~999인, 'large': 1000인 이상)

    Returns:
        dict: 계산 결과 (입력정보, 근로자공제내역, 회사부담내역, 실수령액)
    """
    calc = WageCalculator()

    if salary_type == '연봉':
        result = calc.calculate_from_annual(
            annual_salary=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size
        )
    else:  # 월급
        result = calc.calculate_from_monthly(
            monthly_salary=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size
        )

    return result


def calculate_insurance(
    monthly_income: int,
    non_taxable: int = 0,
    company_size_code: str = 'UNDER_150',
    industry_code: str = 'OTHERS'
) -> dict:
    """
    4대보험료 계산 (국민연금, 건강보험, 장기요양보험, 고용보험, 산재보험)

    Args:
        monthly_income: 월 소득 (원)
        non_taxable: 비과세소득 (원)
        company_size_code: 회사 규모 코드
            - 'UNDER_150': 150인 미만
            - 'PRIORITY_SUPPORT': 150인 이상 우선지원대상기업
            - 'FROM_150_TO_999': 150~999인
            - 'OVER_1000': 1000인 이상
        industry_code: 업종 코드 (산재보험 요율)
            - 'OTHERS': 기타사업 (0.9%)
            - 'WHOLESALE_RETAIL': 도소매/음식/숙박 (0.7%)
            - 'PROFESSIONAL': 전문/과학/기술 (0.5%)
            - 'FINANCE_INSURANCE': 금융/보험 (0.5%)
            - 'CONSTRUCTION': 건설업 (3.5%)
            - 기타: IndustryType enum 참고

    Returns:
        dict: 보험료 상세 내역 (입력정보, 각 보험별 내역, 합계)
    """
    calc = InsuranceCalculator()

    # Convert string codes to enums
    size_map = {
        'UNDER_150': CompanySize.UNDER_150,
        'PRIORITY_SUPPORT': CompanySize.PRIORITY_SUPPORT,
        'FROM_150_TO_999': CompanySize.FROM_150_TO_999,
        'OVER_1000': CompanySize.OVER_1000
    }
    company_size = size_map.get(company_size_code, CompanySize.UNDER_150)

    # Find industry type
    industry = IndustryType.OTHERS
    for ind in IndustryType:
        if ind.name == industry_code:
            industry = ind
            break

    result = calc.calculate_all(
        monthly_income=monthly_income,
        non_taxable=non_taxable,
        company_size=company_size,
        industry=industry
    )

    return result


# GPT Function definitions for Function Calling
CALCULATOR_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_wage",
            "description": "한국 노동법에 따른 임금 계산 (실수령액, 4대보험료, 소득세, 지방소득세). 연봉이나 월급을 입력하면 실제 받게 되는 급여를 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "salary_type": {
                        "type": "string",
                        "enum": ["연봉", "월급"],
                        "description": "급여 유형"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "급여액 (원 단위)"
                    },
                    "tax_free_monthly": {
                        "type": "integer",
                        "description": "월 비과세액 (식대 등, 최대 200,000원)",
                        "default": 0
                    },
                    "dependents": {
                        "type": "integer",
                        "description": "부양가족 수 (본인 포함)",
                        "default": 1
                    },
                    "children_8_to_20": {
                        "type": "integer",
                        "description": "8~20세 자녀 수 (자녀세액공제 대상)",
                        "default": 0
                    },
                    "company_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "회사 규모: small(150인 미만), medium(150~999인), large(1000인 이상)",
                        "default": "small"
                    }
                },
                "required": ["salary_type", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_insurance",
            "description": "한국 4대보험료 상세 계산 (국민연금, 건강보험, 장기요양보험, 고용보험, 산재보험). 근로자와 사업주가 각각 부담하는 보험료를 업종과 회사 규모에 따라 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_income": {
                        "type": "integer",
                        "description": "월 소득 (원 단위)"
                    },
                    "non_taxable": {
                        "type": "integer",
                        "description": "비과세소득 (원 단위)",
                        "default": 0
                    },
                    "company_size_code": {
                        "type": "string",
                        "enum": ["UNDER_150", "PRIORITY_SUPPORT", "FROM_150_TO_999", "OVER_1000"],
                        "description": "회사 규모: UNDER_150(150인 미만), PRIORITY_SUPPORT(우선지원대상), FROM_150_TO_999(150~999인), OVER_1000(1000인 이상)",
                        "default": "UNDER_150"
                    },
                    "industry_code": {
                        "type": "string",
                        "description": "산재보험 업종 코드 (예: OTHERS-기타사업, WHOLESALE_RETAIL-도소매, PROFESSIONAL-전문서비스, FINANCE_INSURANCE-금융, CONSTRUCTION-건설)",
                        "default": "OTHERS"
                    }
                },
                "required": ["monthly_income"]
            }
        }
    }
]


def _build_ncs_filter(query: str) -> dict:
    """Build Pinecone metadata filter from NCS-related query patterns."""
    import re
    filters = {}

    # Detect NCS category mentions
    ncs_categories = {
        '반도체개발': ['반도체 개발', '반도체개발', '제품 기획', '아키텍처', '회로 설계'],
        '반도체장비': ['반도체 장비', '반도체장비', '장비'],
        '반도체재료': ['반도체 재료', '반도체재료', '재료'],
        '반도체제조': ['반도체 제조', '반도체제조', '제조 공정'],
    }
    for cat, keywords in ncs_categories.items():
        for kw in keywords:
            if kw in query:
                filters['ncs_category'] = cat
                break
        if 'ncs_category' in filters:
            break

    # Detect NCS section type from query
    section_patterns = {
        'required_knowledge': ['필요 지식', '필요지식', '이론적 배경', '개념 설명'],
        'performance_procedure': ['수행 순서', '수행순서', '작업 절차', '실습 절차'],
        'performance_content': ['수행 내용', '수행내용'],
        'performance_tip': ['수행 tip', '수행 팁', '실무 팁'],
        'learning_objective': ['학습 목표', '학습목표'],
        'evaluation_criteria': ['평가 준거', '평가준거', '평가 기준'],
        'evaluation_method': ['평가 방법', '평가방법'],
        'feedback': ['피드백'],
        'teaching_method': ['교수 방법', '교수방법', '가르치는 방법'],
        'learning_method': ['학습 방법', '학습방법', '공부 방법'],
        'safety_notes': ['안전', '유의사항', '주의사항'],
        'key_terms': ['핵심 용어', '용어 정리', '핵심용어'],
        'prerequisite': ['선수학습', '선수 학습', '사전 지식'],
        'equipment': ['장비', '공구', '기기'],
    }
    for section_type, keywords in section_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['ncs_section_type'] = section_type
                break
        if 'ncs_section_type' in filters:
            break

    # Detect learning unit mention
    lu_match = re.search(r'학습\s*(\d+)', query)
    if lu_match:
        filters['learning_unit'] = int(lu_match.group(1))

    return filters if filters else None


def parse_mentions(query):
    """Parse @mentions from query and return (clean_query, filters).

    Supports:
    - @파일명.md - specific file
    - @폴더명/ - folder path (ends with /)
    - @키워드 - partial match on source_file
    """
    mentions = re.findall(r'@([^\s@]+)', query)
    clean_query = re.sub(r'@[^\s@]+', '', query).strip()

    filters = []
    for mention in mentions:
        if mention.endswith('/'):
            # Folder filter
            filters.append({'type': 'folder', 'value': mention.rstrip('/')})
        elif '.' in mention:
            # File filter (has extension)
            filters.append({'type': 'file', 'value': mention})
        else:
            # Keyword filter
            filters.append({'type': 'keyword', 'value': mention})

    return clean_query, filters


def build_source_filter(filters):
    """Build Pinecone filter from parsed mentions.

    Note: Pinecone doesn't support substring matching directly,
    so we'll filter results post-query.
    """
    # For now, return None and do post-filtering
    # Pinecone filter would require exact match or $in operator
    return None


# Domain configurations
DOMAIN_CONFIG = {
    'semiconductor': {
        'title': '반도체 AI 검색',
        'icon': '🔬',
        'namespace': '',  # default namespace
        'color': '#00d4ff',
        'color_rgb': '0, 212, 255',
        'gradient_from': '#00d4ff',
        'gradient_to': '#0099cc',
        'description': '반도체 기술 문서 기반 AI 질의응답',
        'sample_questions': [
            'CVD와 PVD 공정의 차이점은 무엇인가요?',
            '반도체 패키징 공정에 대해 설명해주세요',
            '웨이퍼 제조 과정의 주요 단계는?'
        ],
        'features': ['공정 기술', '아키텍처', '제조 공정', '품질 관리']
    },
    'laborlaw': {
        'title': '노동법 AI 상담',
        'icon': '⚖️',
        'namespace': 'laborlaw',
        'color': '#ff9800',
        'color_rgb': '255, 152, 0',
        'gradient_from': '#ff9800',
        'gradient_to': '#f57c00',
        'description': '노동법 및 임금·보험료 계산 전문 AI',
        'sample_questions': [
            '연봉 5000만원 실수령액은 얼마인가요?',
            '주휴수당 계산 방법을 알려주세요',
            '부당해고 판단 기준은 무엇인가요?',
            '4대보험료는 어떻게 계산하나요?'
        ],
        'features': ['임금 계산', '4대보험', '근로기준법', '판례 검색', '법률 상담']
    },
    'msds': {
        'title': 'MSDS 화학물질 정보',
        'icon': '🧪',
        'namespace': 'msds',
        'color': '#4caf50',
        'color_rgb': '76, 175, 80',
        'gradient_from': '#4caf50',
        'gradient_to': '#388e3c',
        'description': '산업안전보건공단 물질안전보건자료 검색',
        'sample_questions': [
            '벤젠의 안전 정보를 알려주세요',
            '에탄올의 유해성·위험성은?',
            '톨루엔의 응급조치요령을 찾아줘',
            'CAS 번호로 화학물질 검색'
        ],
        'features': ['화학물질 검색', 'MSDS 정보', '안전 데이터', 'CAS 번호 조회']
    }
}


@app.route('/')
def home():
    """Home page with domain selection."""
    return render_template('home.html')


@app.route('/semiconductor')
def semiconductor():
    """Semiconductor domain page."""
    config = DOMAIN_CONFIG['semiconductor']
    return render_template('domain.html', domain='semiconductor', config=config)


@app.route('/laborlaw')
def laborlaw():
    """Labor law domain page."""
    config = DOMAIN_CONFIG['laborlaw']
    return render_template('domain.html', domain='laborlaw', config=config)


@app.route('/msds')
def msds():
    """MSDS chemical information page."""
    config = DOMAIN_CONFIG['msds']
    return render_template('msds.html', domain='msds', config=config)


@app.route('/documents/<path:filepath>')
def serve_document(filepath):
    """Serve document images and files."""
    # Normalize unicode for macOS (NFD filesystem) compatibility
    filepath = unicodedata.normalize('NFC', filepath)
    # Try NFC first, fall back to NFD if not found
    full_path = DOCUMENTS_PATH / filepath
    if not full_path.exists():
        filepath_nfd = unicodedata.normalize('NFD', filepath)
        full_path_nfd = DOCUMENTS_PATH / filepath_nfd
        if full_path_nfd.exists():
            filepath = filepath_nfd
    return send_from_directory(DOCUMENTS_PATH, filepath)


def find_related_images(source_file: str) -> list:
    """Find related images from the same document folder."""
    images = []
    try:
        # Normalize unicode for consistent path matching
        normalized_file = unicodedata.normalize('NFC', source_file)
        # Extract the folder path from source_file
        # e.g., "ncs/반도체개발/LM1903060102_23v5_반도체_아키텍처_설계/..." -> folder path
        parts = normalized_file.split('/')
        if len(parts) >= 3:
            folder_path = DOCUMENTS_PATH / parts[0] / parts[1] / parts[2]
            # Try NFD form if NFC path doesn't exist (macOS filesystem)
            if not folder_path.exists():
                nfd_file = unicodedata.normalize('NFD', source_file)
                nfd_parts = nfd_file.split('/')
                folder_path = DOCUMENTS_PATH / nfd_parts[0] / nfd_parts[1] / nfd_parts[2]
            if folder_path.exists():
                # Find all image files in the folder
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    for img_file in folder_path.glob(ext):
                        rel_path = img_file.relative_to(DOCUMENTS_PATH)
                        # Use POSIX path format for URL and normalize to NFC
                        url_path = unicodedata.normalize('NFC', rel_path.as_posix())
                        images.append({
                            'path': f'/documents/{url_path}',
                            'name': img_file.name
                        })
    except Exception as e:
        logging.warning(f"Error finding images: {e}")
    return images[:10]  # Limit to 10 images


@app.route('/api/stats')
def api_stats():
    """Get index statistics."""
    try:
        uploader = get_uploader()
        stats = uploader.get_stats()

        # Format namespaces for frontend
        namespaces = []
        if stats.get('namespaces'):
            for ns_name, ns_info in stats['namespaces'].items():
                namespaces.append({
                    'name': ns_name if ns_name else '(기본)',
                    'vector_count': ns_info.vector_count
                })

        return jsonify({
            'success': True,
            'data': {
                'index_name': os.getenv("PINECONE_INDEX_NAME", "document-index"),
                'dimension': stats.get('dimension', 0),
                'total_vectors': stats.get('total_vector_count', 0),
                'namespaces': namespaces
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for similar content with optional hybrid/keyword modes."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        try:
            top_k = max(1, min(int(data.get('top_k', 5)), 100))
        except (ValueError, TypeError):
            top_k = 5
        namespace = data.get('namespace', '')
        file_type = data.get('file_type', '')
        search_mode = data.get('search_mode', 'vector')  # vector | hybrid | keyword

        # 3-level metadata filters
        domain = data.get('domain', '')
        category = data.get('category', '')
        subcategory = data.get('subcategory', '')

        if not query:
            return jsonify({'success': False, 'error': '검색어를 입력해주세요.'})

        agent = get_agent()

        # Build filter from file_type + metadata levels
        filter_dict = {}
        if file_type:
            filter_dict["file_type"] = file_type
        if domain:
            filter_dict["domain"] = domain
        if category:
            filter_dict["category"] = category
        if subcategory:
            filter_dict["subcategory"] = subcategory
        filter_dict = filter_dict or None

        # Always start with vector search to get candidate documents
        vector_results = agent.search(
            query=query,
            top_k=max(top_k, 20) if search_mode != 'vector' else top_k,
            namespace=namespace,
            filter=filter_dict
        )

        if search_mode == 'vector':
            # Pure vector search - existing behavior
            formatted_results = []
            for r in vector_results:
                metadata = r.get('metadata', {})
                formatted_results.append({
                    'score': round(r.get('score', 0), 4),
                    'source_file': metadata.get('source_file', 'N/A'),
                    'file_type': metadata.get('file_type', 'N/A'),
                    'content': metadata.get('content_preview', metadata.get('content', '')[:500]),
                    'filename': metadata.get('filename', ''),
                    'relative_path': metadata.get('relative_path', ''),
                    'ncs_category': metadata.get('ncs_category', ''),
                    'ncs_document_title': metadata.get('ncs_document_title', ''),
                    'ncs_section_type': metadata.get('ncs_section_type', ''),
                    'ncs_code': metadata.get('ncs_code', ''),
                })
        else:
            # hybrid or keyword mode - use HybridSearcher
            hybrid_searcher = get_hybrid_searcher_instance()
            hybrid_results = hybrid_searcher.search(
                query=query,
                vector_results=vector_results,
                top_k=top_k,
                build_index=True
            )

            if search_mode == 'keyword':
                # Re-sort by BM25 rank (keyword relevance)
                hybrid_results.sort(
                    key=lambda x: x.get('bm25_rank') or 9999
                )
                hybrid_results = hybrid_results[:top_k]

            formatted_results = []
            for r in hybrid_results:
                metadata = r.get('metadata', {})
                bm25_score_val = None
                if hasattr(hybrid_searcher, 'bm25_index') and hybrid_searcher.bm25_index:
                    # Calculate BM25 score for display
                    query_tokens = hybrid_searcher._tokenize(query)
                    scores = hybrid_searcher.bm25_index.get_scores(query_tokens)
                    content = metadata.get('content', '')
                    for idx, doc in hybrid_searcher.doc_map.items():
                        if doc.get('metadata', {}).get('content', '')[:200] == content[:200]:
                            bm25_score_val = round(scores[idx], 4)
                            break

                formatted_results.append({
                    'score': round(r.get('score', 0), 4),
                    'rrf_score': round(r.get('rrf_score', 0), 6),
                    'vector_rank': r.get('vector_rank'),
                    'bm25_rank': r.get('bm25_rank'),
                    'bm25_score': bm25_score_val,
                    'source_file': metadata.get('source_file', 'N/A'),
                    'file_type': metadata.get('file_type', 'N/A'),
                    'content': metadata.get('content_preview', metadata.get('content', '')[:500]),
                    'filename': metadata.get('filename', ''),
                    'relative_path': metadata.get('relative_path', ''),
                    'ncs_category': metadata.get('ncs_category', ''),
                    'ncs_document_title': metadata.get('ncs_document_title', ''),
                    'ncs_section_type': metadata.get('ncs_section_type', ''),
                    'ncs_code': metadata.get('ncs_code', ''),
                })

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'search_mode': search_mode,
                'count': len(formatted_results),
                'results': formatted_results
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def _run_rag_pipeline(data):
    """Shared RAG search pipeline (Phase 1-7). Returns dict with context, sources, messages, tools, etc.
    Raises ValueError if query is empty. Returns early dict with 'answer' key if no results found."""
    query = data.get('query', '').strip()
    namespace = data.get('namespace', '')
    try:
        top_k = max(1, min(int(data.get('top_k', 10)), 100))
    except (ValueError, TypeError):
        top_k = 10
    use_enhancement = data.get('use_enhancement', True)

    logging.info(f"[API/ask] Query: {query[:50]}..., Namespace: {namespace}, TopK: {top_k}, Enhanced: {use_enhancement}")

    if not query:
        raise ValueError('질문을 입력해주세요.')

    # Parse @mentions for source filtering
    clean_query, mention_filters = parse_mentions(query)

    # Build search query
    if mention_filters:
        filter_keywords = ' '.join([f['value'].replace('_', ' ') for f in mention_filters])
        if clean_query and len(clean_query) >= 3:
            search_query = f"{filter_keywords} {clean_query}"
        else:
            search_query = filter_keywords
    else:
        search_query = clean_query if clean_query else query

    agent = get_agent()
    client = get_openai_client()

    # ========================================
    # Phase 1: Query Enhancement
    # ========================================
    enhanced_queries = [search_query]
    keywords = []

    if use_enhancement:
        try:
            query_enhancer = get_query_enhancer()

            # Generate query variations for broader retrieval
            enhanced_queries = query_enhancer.multi_query(search_query, num_variations=2)
            logging.info(f"[Query Enhancement] Generated {len(enhanced_queries)} query variations")

            # HyDE: generate hypothetical document for longer queries
            if len(search_query) >= 30:
                try:
                    domain = 'laborlaw' if namespace == 'laborlaw' else 'semiconductor'
                    hyde_doc = query_enhancer.hyde(search_query, domain=domain)
                    if hyde_doc and hyde_doc != search_query:
                        enhanced_queries.append(hyde_doc)
                        logging.info(f"[HyDE] Added hypothetical document query")
                except Exception as e:
                    logging.warning(f"[HyDE] Failed: {e}")

            # Extract keywords for potential filtering
            keywords = query_enhancer.extract_keywords(search_query)
            logging.info(f"[Query Enhancement] Keywords: {keywords}")

        except Exception as e:
            logging.warning(f"[Query Enhancement] Failed: {e}, using original query")
            enhanced_queries = [search_query]

    # ========================================
    # Phase 2: Multi-Query Search (with NCS metadata filtering)
    # ========================================
    # Build NCS-aware metadata filter from query
    ncs_filter = _build_ncs_filter(search_query)

    # Search with multiple query variations and merge results
    # When BM25 is skipped, fetch more candidates so the reranker has a wider pool
    skip_bm25_env = os.environ.get('SKIP_BM25_HYBRID', '').lower() in ('true', '1', 'yes')
    if mention_filters:
        search_top_k = top_k * 3
    elif skip_bm25_env:
        search_top_k = top_k * 4  # Wider recall when relying on reranker alone
    else:
        search_top_k = top_k * 2  # Fetch more for filtering/reranking
    is_all_namespace = (namespace == 'all')
    all_results = []
    seen_ids = set()

    for eq in enhanced_queries:
        try:
            if is_all_namespace:
                # Multi-namespace simultaneous query via Pinecone server-side parallelism
                try:
                    uploader = get_uploader()
                    stats = uploader.get_stats()
                    ns_list = [ns for ns in stats.get('namespaces', {}).keys() if ns]
                except Exception:
                    ns_list = ['semiconductor', 'laborlaw', 'field-training']
                results = agent.search_all_namespaces(
                    query=eq,
                    namespaces=ns_list,
                    top_k=search_top_k,
                    filter=ncs_filter
                )
            else:
                results = agent.search(
                    query=eq,
                    top_k=search_top_k,
                    namespace=namespace,
                    filter=ncs_filter
                )
            for r in results:
                # Deduplicate by content hash
                content = r.get('metadata', {}).get('content', '')
                content_id = hash(content[:200])
                if content_id not in seen_ids:
                    seen_ids.add(content_id)
                    all_results.append(r)
        except Exception as e:
            logging.warning(f"[Search] Failed for query '{eq[:30]}...': {e}")

    results = all_results
    logging.info(f"[Search] Retrieved {len(results)} unique documents from {len(enhanced_queries)} queries")

    # ========================================
    # Phase 3: Apply mention filters
    # ========================================
    if mention_filters and results:
        filtered_results = []
        for r in results:
            source_file = unicodedata.normalize('NFC', r.get('metadata', {}).get('source_file', ''))
            filename = unicodedata.normalize('NFC', r.get('metadata', {}).get('filename', ''))

            match = False
            for f in mention_filters:
                filter_value = unicodedata.normalize('NFC', f['value'].lower())
                if f['type'] == 'file':
                    if filter_value in filename.lower():
                        match = True
                        break
                elif f['type'] == 'folder':
                    if filter_value in source_file.lower():
                        match = True
                        break
                elif f['type'] == 'keyword':
                    if filter_value in source_file.lower():
                        match = True
                        break

            if match:
                filtered_results.append(r)

        results = filtered_results
        logging.info(f"[Filter] After mention filtering: {len(results)} documents")

    if not results:
        return {
            'early_response': True,
            'data': {
                'answer': '관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해주세요.',
                'sources': [],
                'enhancement_used': use_enhancement
            }
        }

    # ========================================
    # Phase 4: Hybrid Search (BM25 + Vector)
    # ========================================
    skip_bm25 = os.environ.get('SKIP_BM25_HYBRID', '').lower() in ('true', '1', 'yes')
    if skip_bm25:
        logging.info("[Hybrid Search] Skipped (SKIP_BM25_HYBRID=true)")
    elif use_enhancement and len(results) > 3:
        try:
            hybrid_searcher = get_hybrid_searcher_instance()
            results = hybrid_searcher.search_with_keyword_boost(
                query=search_query,
                vector_results=results,
                keywords=keywords,
                top_k=min(len(results), top_k * 2)
            )
            logging.info(f"[Hybrid Search] Applied BM25 + keyword boosting")
        except Exception as e:
            logging.warning(f"[Hybrid Search] Failed: {e}, using vector results only")

    # ========================================
    # Phase 5: Reranking
    # ========================================
    if use_enhancement and len(results) > 3:
        try:
            reranker = get_reranker_instance()
            results = reranker.rerank(
                query=search_query,
                docs=results,
                top_k=min(len(results), top_k + 5)  # Keep extra for context optimization
            )
            logging.info(f"[Reranking] Reranked to {len(results)} documents")
        except Exception as e:
            logging.warning(f"[Reranking] Failed: {e}, using original order")

    # ========================================
    # Phase 6: Context Optimization
    # ========================================
    if use_enhancement:
        try:
            context_optimizer = get_context_optimizer()

            # Deduplicate similar content
            results = context_optimizer.deduplicate(results)

            # Reorder for LLM attention (Lost in the Middle prevention)
            results = context_optimizer.reorder_for_llm(results, strategy="lost_in_middle")

            logging.info(f"[Context Optimization] Final context: {len(results)} documents")
        except Exception as e:
            logging.warning(f"[Context Optimization] Failed: {e}")

    # Filter by minimum relevance score
    MIN_RELEVANCE_SCORE = 0.3
    results = [r for r in results if r.get('rerank_score', r.get('rrf_score', r.get('score', 0))) >= MIN_RELEVANCE_SCORE]

    # Limit to top_k after all processing
    results = results[:top_k]

    # ========================================
    # Phase 7: Build Context and Sources
    # ========================================
    context_parts = []
    sources = []

    for i, r in enumerate(results):
        metadata = r.get('metadata', {})
        content = metadata.get('content', '')
        source_file = metadata.get('source_file', 'Unknown')
        file_type = metadata.get('file_type', 'unknown')
        score = r.get('rerank_score', r.get('rrf_score', r.get('score', 0)))

        if content:
            context_parts.append(f"[문서 {i+1}] (출처: {source_file})\n{content}")

            # Build image URL for image files (normalize unicode for macOS)
            image_url = None
            if file_type == 'image' or source_file.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                image_url = f'/documents/{unicodedata.normalize("NFC", source_file)}'

            sources.append({
                'source_file': source_file,
                'file_type': file_type,
                'score': round(score, 4) if isinstance(score, float) else score,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'image_url': image_url,
                'ncs_category': metadata.get('ncs_category', ''),
                'ncs_document_title': metadata.get('ncs_document_title', ''),
                'ncs_section_type': metadata.get('ncs_section_type', ''),
                'ncs_code': metadata.get('ncs_code', ''),
            })

    context = "\n\n---\n\n".join(context_parts)

    return {
        'early_response': False,
        'query': query,
        'namespace': namespace,
        'search_query': search_query,
        'context': context,
        'sources': sources,
        'context_parts': context_parts,
        'enhanced_queries': enhanced_queries,
        'keywords': keywords,
        'use_enhancement': use_enhancement,
        'client': client,
    }


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """RAG endpoint - search and generate comprehensive answer with enhanced retrieval."""
    try:
        data = request.get_json()
        pipeline = _run_rag_pipeline(data)

        if pipeline.get('early_response'):
            return jsonify({'success': True, 'data': pipeline['data']})

        query = pipeline['query']
        namespace = pipeline['namespace']
        context = pipeline['context']
        sources = pipeline['sources']
        enhanced_queries = pipeline['enhanced_queries']
        keywords = pipeline['keywords']
        use_enhancement = pipeline['use_enhancement']
        client = pipeline['client']

        # ========================================
        # Phase 8: Generate Answer with Improved Prompt
        # ========================================
        # Select domain-specific prompt based on namespace
        base_prompt = DOMAIN_PROMPTS.get(namespace, DEFAULT_SYSTEM_PROMPT)

        # Enhanced Chain-of-Thought prompt additions
        cot_instructions = """

## 답변 생성 프로세스

1. **이해 단계**: 질문의 핵심 의도를 파악하세요
2. **분석 단계**: 제공된 문서에서 관련 정보를 체계적으로 추출하세요
3. **종합 단계**: 추출된 정보를 논리적으로 구조화하세요
4. **검증 단계**: 답변이 질문에 완전히 답하는지 확인하세요

## 답변 구조화 지침

- **핵심 요약**: 먼저 1-2문장으로 핵심 답변 제시
- **상세 설명**: 관련 개념, 원리, 프로세스 설명
- **구체적 예시**: 가능한 경우 실제 사례나 수치 제시
- **추가 정보**: 관련된 보충 정보나 주의사항"""

        # Common visual representation guidelines (appended to all prompts)
        visual_guidelines = """

## 시각적 표현 지침

10. **비교 정보**는 반드시 표(table)로 정리하세요
   예시:
   | 구분 | 항목1 | 항목2 |
   |------|-------|-------|
   | 특징 | 설명1 | 설명2 |

11. **프로세스/단계**는 순서 목록으로 정리하세요
12. **수치 데이터**가 있으면 정리해서 제시하세요 (차트로 시각화할 수 있도록)
13. **분류/종류**는 표나 목록으로 체계적으로 정리하세요

데이터 시각화 요청시:
- 수치 비교가 필요한 경우, 다음 JSON 형식으로 차트 데이터를 추가로 제공하세요:
<!--CHART_DATA
{
  "type": "bar|pie|line",
  "title": "차트 제목",
  "labels": ["라벨1", "라벨2"],
  "data": [수치1, 수치2],
  "unit": "단위"
}
CHART_DATA-->"""

        system_prompt = base_prompt + cot_instructions + visual_guidelines

        user_prompt = f"""## 질문
{query}

## 참고 문서 ({len(sources)}개)
{context}

위 문서들을 참고하여 질문에 대해 종합적으로 답변해주세요.

**중요 지침:**
1. 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 핵심 답변을 먼저 제시하고, 상세 설명을 이어가세요
4. 기술 용어는 한글과 영문을 병기하세요"""

        # Enable calculator functions only for laborlaw namespace
        tools = CALCULATOR_FUNCTIONS if namespace == 'laborlaw' else None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Initial GPT call (with function calling if tools available)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.3,
            max_tokens=2500  # Increased for more detailed responses
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Handle function calls
        calculation_results = []
        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name

                # Parse function arguments with error handling
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logging.error(f"[Function Call] Invalid JSON args for {function_name}: {e}")
                    logging.error(f"[Function Call] Raw arguments: {tool_call.function.arguments}")
                    function_response = {"error": f"잘못된 함수 인자 형식: {str(e)}"}
                    calculation_results.append({
                        'function': function_name,
                        'args': {},
                        'result': function_response
                    })
                    # Add error response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response, ensure_ascii=False, indent=2)
                    })
                    continue

                logging.info(f"[Function Call] {function_name} with args: {function_args}")

                # Execute the function
                try:
                    if function_name == "calculate_wage":
                        function_response = calculate_wage(**function_args)
                    elif function_name == "calculate_insurance":
                        function_response = calculate_insurance(**function_args)
                    else:
                        function_response = {"error": "Unknown function"}
                except Exception as e:
                    logging.error(f"[Function Call] Execution error in {function_name}: {e}")
                    function_response = {"error": f"함수 실행 오류: {str(e)}"}

                calculation_results.append({
                    'function': function_name,
                    'args': function_args,
                    'result': function_response
                })

                # Add function response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(function_response, ensure_ascii=False, indent=2)
                })

            # Get final response with function results
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=2500
            )

            answer = second_response.choices[0].message.content
        else:
            answer = response_message.content

        # Collect related images from source documents
        related_images = []
        seen_folders = set()
        for source in sources:
            source_file = source.get('source_file', '')
            # Extract folder path to avoid duplicates
            folder_key = '/'.join(source_file.split('/')[:3])
            if folder_key not in seen_folders:
                seen_folders.add(folder_key)
                images = find_related_images(source_file)
                related_images.extend(images)

        # Limit total images
        related_images = related_images[:12]

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'answer': answer,
                'sources': sources,
                'source_count': len(sources),
                'images': related_images,
                'calculations': calculation_results if calculation_results else None,
                'enhancement_used': use_enhancement,
                'query_variations': enhanced_queries if use_enhancement else None,
                'keywords_extracted': keywords if use_enhancement else None
            }
        })

    except Exception as e:
        logging.exception(f"[API/ask] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def _build_llm_messages(query, sources, context, namespace):
    """Build system prompt and user message for LLM call."""
    base_prompt = DOMAIN_PROMPTS.get(namespace, DEFAULT_SYSTEM_PROMPT)

    cot_instructions = """

## 답변 생성 프로세스

1. **이해 단계**: 질문의 핵심 의도를 파악하세요
2. **분석 단계**: 제공된 문서에서 관련 정보를 체계적으로 추출하세요
3. **종합 단계**: 추출된 정보를 논리적으로 구조화하세요
4. **검증 단계**: 답변이 질문에 완전히 답하는지 확인하세요

## 답변 구조화 지침

- **핵심 요약**: 먼저 1-2문장으로 핵심 답변 제시
- **상세 설명**: 관련 개념, 원리, 프로세스 설명
- **구체적 예시**: 가능한 경우 실제 사례나 수치 제시
- **추가 정보**: 관련된 보충 정보나 주의사항"""

    visual_guidelines = """

## 시각적 표현 지침

10. **비교 정보**는 반드시 표(table)로 정리하세요
   예시:
   | 구분 | 항목1 | 항목2 |
   |------|-------|-------|
   | 특징 | 설명1 | 설명2 |

11. **프로세스/단계**는 순서 목록으로 정리하세요
12. **수치 데이터**가 있으면 정리해서 제시하세요 (차트로 시각화할 수 있도록)
13. **분류/종류**는 표나 목록으로 체계적으로 정리하세요

데이터 시각화 요청시:
- 수치 비교가 필요한 경우, 다음 JSON 형식으로 차트 데이터를 추가로 제공하세요:
<!--CHART_DATA
{
  "type": "bar|pie|line",
  "title": "차트 제목",
  "labels": ["라벨1", "라벨2"],
  "data": [수치1, 수치2],
  "unit": "단위"
}
CHART_DATA-->"""

    system_prompt = base_prompt + cot_instructions + visual_guidelines

    user_prompt = f"""## 질문
{query}

## 참고 문서 ({len(sources)}개)
{context}

위 문서들을 참고하여 질문에 대해 종합적으로 답변해주세요.

**중요 지침:**
1. 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 핵심 답변을 먼저 제시하고, 상세 설명을 이어가세요
4. 기술 용어는 한글과 영문을 병기하세요"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _handle_tool_calls(client, messages, tool_calls):
    """Execute tool calls and return (calculation_results, updated_messages)."""
    calculation_results = []
    messages.append(tool_calls)  # append the assistant message with tool_calls

    for tool_call in tool_calls.tool_calls:
        function_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logging.error(f"[Function Call] Invalid JSON args for {function_name}: {e}")
            function_response = {"error": f"잘못된 함수 인자 형식: {str(e)}"}
            calculation_results.append({'function': function_name, 'args': {}, 'result': function_response})
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False, indent=2)
            })
            continue

        logging.info(f"[Function Call] {function_name} with args: {function_args}")
        try:
            if function_name == "calculate_wage":
                function_response = calculate_wage(**function_args)
            elif function_name == "calculate_insurance":
                function_response = calculate_insurance(**function_args)
            else:
                function_response = {"error": "Unknown function"}
        except Exception as e:
            logging.error(f"[Function Call] Execution error in {function_name}: {e}")
            function_response = {"error": f"함수 실행 오류: {str(e)}"}

        calculation_results.append({'function': function_name, 'args': function_args, 'result': function_response})
        messages.append({
            "role": "tool", "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_response, ensure_ascii=False, indent=2)
        })

    return calculation_results, messages


@app.route('/api/ask/stream', methods=['POST'])
def api_ask_stream():
    """SSE streaming RAG endpoint - sends metadata first, then streams LLM answer token by token."""
    try:
        data = request.get_json()
        pipeline = _run_rag_pipeline(data)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        logging.exception(f"[API/ask/stream] Pipeline error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

    if pipeline.get('early_response'):
        return jsonify({'success': True, 'data': pipeline['data']})

    query = pipeline['query']
    namespace = pipeline['namespace']
    context = pipeline['context']
    sources = pipeline['sources']
    enhanced_queries = pipeline['enhanced_queries']
    keywords = pipeline['keywords']
    use_enhancement = pipeline['use_enhancement']
    client = pipeline['client']

    messages = _build_llm_messages(query, sources, context, namespace)
    tools = CALCULATOR_FUNCTIONS if namespace == 'laborlaw' else None

    def generate():
        try:
            calculation_results = []

            # Collect related images
            related_images = []
            seen_folders = set()
            for source in sources:
                source_file = source.get('source_file', '')
                folder_key = '/'.join(source_file.split('/')[:3])
                if folder_key not in seen_folders:
                    seen_folders.add(folder_key)
                    images = find_related_images(source_file)
                    related_images.extend(images)
            related_images = related_images[:12]

            # Send metadata event first (sources, images, query info)
            metadata_event = json.dumps({
                'type': 'metadata',
                'data': {
                    'query': query,
                    'sources': sources,
                    'source_count': len(sources),
                    'images': related_images,
                    'enhancement_used': use_enhancement,
                    'query_variations': enhanced_queries if use_enhancement else None,
                    'keywords_extracted': keywords if use_enhancement else None
                }
            }, ensure_ascii=False)
            yield f"data: {metadata_event}\n\n"

            # If tools available, first call is non-streaming to check for tool_calls
            if tools:
                first_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=2500
                )
                response_message = first_response.choices[0].message

                if response_message.tool_calls:
                    calc_results, messages_updated = _handle_tool_calls(client, messages, response_message)
                    calculation_results = calc_results

                    # Send calculation results event
                    calc_event = json.dumps({
                        'type': 'calculations',
                        'data': calculation_results
                    }, ensure_ascii=False)
                    yield f"data: {calc_event}\n\n"

                    # Stream the final response after tool calls
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages_updated,
                        temperature=0.3,
                        max_tokens=2500,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            token_event = json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)
                            yield f"data: {token_event}\n\n"
                else:
                    # No tool calls - stream the content from first response
                    if response_message.content:
                        # First response was non-streaming, send it as tokens
                        token_event = json.dumps({'type': 'token', 'data': response_message.content}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"
            else:
                # No tools - stream directly
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2500,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        token_event = json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"

            # Send done event
            done_event = json.dumps({
                'type': 'done',
                'data': {'calculations': calculation_results if calculation_results else None}
            }, ensure_ascii=False)
            yield f"data: {done_event}\n\n"

        except Exception as e:
            logging.exception(f"[API/ask/stream] Streaming error: {str(e)}")
            error_event = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_event}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/namespaces')
def api_namespaces():
    """Get list of namespaces."""
    try:
        uploader = get_uploader()
        stats = uploader.get_stats()

        namespaces = []
        if stats.get('namespaces'):
            for ns_name in stats['namespaces'].keys():
                namespaces.append(ns_name if ns_name else '')

        return jsonify({
            'success': True,
            'data': namespaces
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sources')
def api_sources():
    """Get list of available source files and folders for autocomplete."""
    try:
        namespace = request.args.get('namespace', '')
        agent = get_agent()

        # Use generic queries based on namespace to find documents
        # Different queries for different document domains
        generic_queries = ["문서", "정보", "내용", "법률", "규정", "기술"]

        all_results = []
        for query in generic_queries[:2]:  # Use top 2 queries to reduce API calls
            results = agent.search(
                query=query,
                top_k=50,
                namespace=namespace
            )
            all_results.extend(results)

        # Deduplicate by source_file
        seen = set()
        results = []
        for r in all_results:
            source_file = r.get('metadata', {}).get('source_file', '')
            if source_file and source_file not in seen:
                seen.add(source_file)
                results.append(r)

        folders = set()
        files = set()

        for r in results:
            metadata = r.get('metadata', {})
            source_file = metadata.get('source_file', '')
            filename = metadata.get('filename', '')

            if filename:
                files.add(filename)

            if source_file:
                # Extract folder paths
                parts = source_file.split('/')
                for i in range(1, len(parts)):
                    folder = '/'.join(parts[:i])
                    if folder:
                        folders.add(folder)

        return jsonify({
            'success': True,
            'data': {
                'folders': sorted(folders),
                'files': sorted(files)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/delete', methods=['POST'])
def api_delete():
    """Delete vectors."""
    try:
        data = request.get_json()
        namespace = data.get('namespace', '')
        delete_all = data.get('delete_all', False)
        source_file = data.get('source_file', '')

        uploader = get_uploader()

        if delete_all:
            uploader.index.delete(delete_all=True, namespace=namespace)
            return jsonify({
                'success': True,
                'message': f"네임스페이스 '{namespace or '(기본)'}' 의 모든 벡터가 삭제되었습니다."
            })
        elif source_file:
            success = uploader.delete_by_filter(
                filter={"source_file": source_file},
                namespace=namespace
            )
            if success:
                return jsonify({
                    'success': True,
                    'message': f"'{source_file}'의 벡터가 삭제되었습니다."
                })
            else:
                return jsonify({'success': False, 'error': '삭제 실패'})
        else:
            return jsonify({'success': False, 'error': '삭제할 대상을 지정해주세요.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# MSDS API Client instance
msds_client = MsdsApiClient()


@app.route('/api/msds/search', methods=['POST'])
def msds_search():
    """Search chemicals in KOSHA MSDS database."""
    try:
        data = request.json
        search_word = data.get('search_word', '')
        search_type = int(data.get('search_type', 0))  # 0=국문명, 1=CAS, 2=UN, 3=KE, 4=EN
        page_no = int(data.get('page_no', 1))
        num_of_rows = int(data.get('num_of_rows', 10))

        if not search_word:
            return jsonify({'success': False, 'error': '검색어를 입력해주세요.'})

        result = msds_client.search_chemicals(
            search_word=search_word,
            search_type=search_type,
            page_no=page_no,
            num_of_rows=num_of_rows
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/msds/detail', methods=['POST'])
def msds_detail():
    """Get detailed chemical information from KOSHA MSDS."""
    try:
        data = request.json
        chem_id = data.get('chem_id', '')
        section = data.get('section', '')  # Optional: specific section (01-16)

        if not chem_id:
            return jsonify({'success': False, 'error': '화학물질 ID를 입력해주세요.'})

        # If section is specified, get that section only
        if section:
            result = msds_client.get_chemical_detail(chem_id, section)
        else:
            # Get all sections
            result = msds_client.get_full_chemical_detail(chem_id)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/msds/identify', methods=['POST'])
def msds_identify():
    """Identify chemical substance from image using OpenAI Vision API."""
    try:
        data = request.json
        image_data = data.get('image', '')  # Base64 encoded image

        if not image_data:
            return jsonify({'success': False, 'error': '이미지 데이터가 없습니다.'})

        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Use OpenAI Vision API to identify chemical
        client = get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """이미지에서 화학물질 정보를 찾아주세요.

다음 정보를 추출해주세요:
1. 화학물질명 (국문명 또는 영문명)
2. CAS 번호 (있는 경우)
3. 제품명 또는 상품명 (있는 경우)

응답은 반드시 다음 JSON 형식으로만 작성해주세요:
{
    "chemical_name": "화학물질명",
    "cas_no": "CAS 번호 (없으면 null)",
    "product_name": "제품명 (없으면 null)"
}

이미지에서 화학물질 관련 정보를 찾을 수 없다면:
{
    "chemical_name": null,
    "cas_no": null,
    "product_name": null,
    "error": "화학물질 정보를 찾을 수 없습니다."
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Parse response
        result_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # If no JSON found, try to parse the whole response
            result = json.loads(result_text)

        # If chemical name found, search for it
        if result.get('chemical_name'):
            search_word = result['chemical_name']
            search_type = 0  # Default to Korean name

            # If CAS number is available, use that for search
            if result.get('cas_no'):
                search_word = result['cas_no']
                search_type = 1  # CAS number search

            # Search in MSDS database
            search_result = msds_client.search_chemicals(
                search_word=search_word,
                search_type=search_type,
                page_no=1,
                num_of_rows=10
            )

            return jsonify({
                'success': True,
                'identified': result,
                'search_result': search_result
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '화학물질 정보를 찾을 수 없습니다.'),
                'identified': result
            })

    except json.JSONDecodeError as e:
        return jsonify({
            'success': False,
            'error': f'응답 파싱 실패: {str(e)}',
            'raw_response': result_text if 'result_text' in locals() else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        exit(1)

    print("🚀 Pinecone Agent Web Interface")
    print("=" * 40)
    print(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'document-index')}")
    print("=" * 40)
    print("\n🌐 http://localhost:5001 에서 접속하세요\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
