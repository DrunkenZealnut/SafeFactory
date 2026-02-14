"""Domain-specific metadata filter builders and mention parsing."""

import re
from typing import Optional


def build_ncs_filter(query: str) -> Optional[dict]:
    """Build Pinecone metadata filter from NCS-related query patterns."""
    if not query:
        return None
    filters = {}

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

    lu_match = re.search(r'학습\s*(\d+)', query)
    if lu_match:
        filters['learning_unit'] = int(lu_match.group(1))

    return filters if filters else None


def build_laborlaw_filter(query: str) -> Optional[dict]:
    """Build Pinecone metadata filter for laborlaw queries."""
    if not query:
        return None
    filters = {}

    content_type_patterns = {
        'law': ['법률', '법조항', '법령', '규정', '조항'],
        'case': ['사례', '판례', '상담', '사건'],
        'qa': ['질의', '회시', 'Q&A', '질문'],
    }
    for ct, keywords in content_type_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['content_type'] = ct
                break
        if 'content_type' in filters:
            break

    category_patterns = {
        'wages': ['임금', '급여', '최저임금', '체불', '금품 청산', '퇴직급여', '실수령'],
        'working_hours': ['근로시간', '연장근로', '야간근로', '휴일', '연차', '휴가', '탄력적', '주휴'],
        'employment_contract': ['근로계약', '해고', '부당해고', '퇴직', '해고 예고', '계약해지'],
        'women_minors': ['여성', '임산부', '모성', '육아', '생리휴가', '출산'],
        'safety_health': ['안전', '산재', '산업재해', '보건'],
        'workplace_harassment': ['괴롭힘', '직장 내 괴롭힘'],
        'social_insurance': ['4대보험', '국민연금', '건강보험', '고용보험', '산재보험'],
        'non_regular_workers': ['파견', '기간제', '비정규', '단시간'],
        'labor_unions': ['노동조합', '단체교섭', '쟁의', '파업'],
        'discrimination': ['차별', '균등처우', '성희롱', '평등'],
        'accident_compensation': ['재해보상', '요양보상', '휴업보상', '장해보상'],
        'enforcement_penalties': ['벌칙', '과태료', '근로감독'],
    }
    for category, keywords in category_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['law_category'] = category
                break
        if 'law_category' in filters:
            break

    law_name_patterns = {
        '근로기준법': ['근로기준법', '근기법'],
        '최저임금법': ['최저임금법'],
        '산업안전보건법': ['산업안전보건법', '산안법'],
        '고용보험법': ['고용보험법'],
        '산업재해보상보험법': ['산재보험법', '산업재해보상보험법'],
        '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률': ['남녀고용평등법', '고용평등법'],
        '파견근로자 보호 등에 관한 법률': ['파견법', '파견근로자보호법'],
        '노동조합 및 노동관계조정법': ['노동조합법', '노조법'],
    }
    for law_name, keywords in law_name_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['law_name'] = law_name
                break
        if 'law_name' in filters:
            break

    article_match = re.search(r'제\d+조(?:의\d+)?', query)
    if article_match:
        filters['article_number'] = article_match.group(0)

    return filters if filters else None


def build_field_training_filter(query: str) -> Optional[dict]:
    """Build Pinecone metadata filter for field-training queries."""
    if not query:
        return None
    filters = {}

    if '카드북' in query:
        filters['training_type'] = 'cardbook'
        cb_match = re.search(r'카드북\s*(\d+)', query)
        if cb_match:
            filters['cardbook_number'] = int(cb_match.group(1))
    elif '건강관리' in query or '건강 관리' in query:
        filters['training_type'] = 'health_guide'

    equipment_patterns = {
        '차량계 건설기계작업': ['차량계', '건설기계', '굴삭기', '지게차', '크레인', '불도저', '로더'],
        '중량물운반': ['중량물', '운반', '하역'],
        '금속성형기계작업': ['금속성형', '프레스', '전단기', '절곡기'],
        '금속절삭기계': ['금속절삭', '연삭기', '선반', '밀링', '절삭기계', '드릴'],
        '식품제조작업': ['식품', '식품제조', '혼합기', '분쇄기'],
        '세척제취급작업': ['세척제', '세척', '유기용제'],
    }
    for equip_type, keywords in equipment_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['equipment_type'] = equip_type
                break
        if 'equipment_type' in filters:
            break

    section_patterns = {
        'characteristics': ['특성', '특징', '구조'],
        'accident_types': ['재해', '사고', '위험 유형', '위험요인'],
        'safety_rules': ['안전수칙', '안전 수칙', '안전조치', '주의사항'],
        'hazard_factors': ['유해요인', '유해물질', '노출'],
        'health_management': ['건강관리', '건강영향', '증상'],
        'process_overview': ['공정', '제조공정', '제조환경'],
        'protective_equipment': ['보호구', '보호장비', '보안경'],
        'msds_info': ['MSDS', '물질안전'],
    }
    for section_type, keywords in section_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['ft_section_type'] = section_type
                break
        if 'ft_section_type' in filters:
            break

    hazard_patterns = {
        'chemical_exposure': ['화학물질', '유기용제', '불산', '암모니아'],
        'cuts': ['베임', '절단'],
        'entanglement': ['끼임', '말림'],
        'struck_by': ['맞음', '비산', '날림'],
        'falls': ['넘어짐', '미끄러짐', '추락'],
        'dust_inhalation': ['분진', '흡입'],
    }
    for hazard, keywords in hazard_patterns.items():
        for kw in keywords:
            if kw in query:
                filters['hazard_category'] = hazard
                break
        if 'hazard_category' in filters:
            break

    return filters if filters else None


def build_domain_filter(query: str, namespace: str) -> Optional[dict]:
    """Build Pinecone metadata filter based on domain namespace and query patterns."""
    if not query:
        return None
    if namespace == 'laborlaw':
        return build_laborlaw_filter(query)
    elif namespace == 'field-training':
        return build_field_training_filter(query)
    elif namespace == 'all':
        return None
    else:
        return build_ncs_filter(query)


def parse_mentions(query):
    """Parse @mentions from query and return (clean_query, filters)."""
    mentions = re.findall(r'@([^\s@]+)', query)
    clean_query = re.sub(r'@[^\s@]+', '', query).strip()

    filters = []
    for mention in mentions:
        if mention.endswith('/'):
            filters.append({'type': 'folder', 'value': mention.rstrip('/')})
        elif '.' in mention:
            filters.append({'type': 'file', 'value': mention})
        else:
            filters.append({'type': 'keyword', 'value': mention})

    return clean_query, filters


def build_source_filter(filters):
    """Build Pinecone filter from parsed mentions (post-filtering only)."""
    # TODO: Implement source filtering logic based on parsed @mention filters
    return None
