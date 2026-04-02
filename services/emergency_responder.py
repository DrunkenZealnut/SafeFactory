"""Emergency intent classifier and fast-track response module.

Detects emergency/urgent queries (chemical exposure, injury, burns, etc.)
and returns pre-written first-aid manuals instantly, bypassing the full RAG pipeline.

Designed for vocational high school students in industrial settings.
"""

import logging
import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Emergency keyword patterns by category
# ---------------------------------------------------------------------------

EMERGENCY_PATTERNS = {
    'chemical_exposure': {
        'keywords': [
            '화학물질', '약품', '쏟았', '쏟아', '튀었', '튀어',
            '눈에 들어', '피부에 묻', '삼켰', '마셨', '흡입',
            '냄새가 심', '가스 누출', '누출', '불산', '염산', '황산',
            '유독', '중독', '약품이 눈', '약품이 손', '약품이 피부',
        ],
        'patterns': [
            re.compile(r'(?:화학|약품|산|알칼리|용제).*(?:쏟|튀|묻|닿|들어갔|접촉)'),
            re.compile(r'(?:눈|피부|손|얼굴).*(?:화학|약품|산|따가|아프|빨갛)'),
        ],
    },
    'injury_bleeding': {
        'keywords': [
            '다쳤', '다쳐', '다쳐서', '베였', '베어', '찢어', '찢겨',
            '피가 나', '피가 안 멈', '출혈', '부러', '골절',
            '끼었', '끼임', '절단', '찍혔', '찔렸', '맞았',
        ],
        'patterns': [
            re.compile(r'(?:손|손가락|팔|다리|발|머리|눈).*(?:다치|베이|찢|부러|끼)'),
            re.compile(r'(?:피|출혈).*(?:나|안 멈|많|심)'),
        ],
    },
    'burn': {
        'keywords': [
            '화상', '데었', '데여', '데임', '뜨거운', '불꽃', '스파크',
            '용접', '증기', '열탕', '화염',
        ],
        'patterns': [
            re.compile(r'(?:뜨거|높은 온도|고온|열|불).*(?:데|닿|튀|접촉)'),
        ],
    },
    'electric_shock': {
        'keywords': [
            '감전', '전기', '누전', '합선', '쇼크', '찌릿', '전류',
        ],
        'patterns': [
            re.compile(r'(?:전기|전선|콘센트|배선).*(?:감전|닿|만지|찌릿)'),
        ],
    },
    'breathing_difficulty': {
        'keywords': [
            '숨이 안', '호흡곤란', '숨을 못', '기침이 심', '어지러',
            '쓰러졌', '의식 잃', '졸도', '질식', '실신', '현기증',
        ],
        'patterns': [
            re.compile(r'(?:숨|호흡|의식).*(?:안 쉬|못 쉬|잃|없)'),
            re.compile(r'(?:어지럽|현기증|졸도|쓰러)'),
        ],
    },
    'fall_crush': {
        'keywords': [
            '추락', '떨어졌', '떨어져', '무거운 것', '깔렸', '매달려',
            '협착', '무너졌', '붕괴',
        ],
        'patterns': [
            re.compile(r'(?:높은 곳|사다리|지붕|비계).*(?:떨어|추락|미끄러)'),
        ],
    },
}

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_emergency(query: str) -> Optional[Tuple[str, float]]:
    """Classify whether a query is an emergency and return its category.

    Uses keyword matching + regex pattern matching (weighted x2).
    Threshold: score >= 1 triggers emergency.

    Args:
        query: User question text.

    Returns:
        (category, confidence) tuple, or None if not emergency.
        confidence is clamped to [0.0, 1.0].
    """
    query_lower = query.lower()
    best_category = None
    best_score = 0

    for category, config in EMERGENCY_PATTERNS.items():
        score = 0
        for kw in config['keywords']:
            if kw in query_lower:
                score += 1
        for pattern in config['patterns']:
            if pattern.search(query_lower):
                score += 2

        if score > best_score:
            best_score = score
            best_category = category

    if best_score >= 1 and best_category:
        confidence = min(best_score / 5.0, 1.0)
        logging.info(
            "[Emergency] Detected: %s (score=%d, conf=%.2f) for '%.40s'",
            best_category, best_score, confidence, query,
        )
        return best_category, confidence

    return None


# ---------------------------------------------------------------------------
# Pre-written emergency manuals (static, no LLM call needed)
# ---------------------------------------------------------------------------

EMERGENCY_MANUALS = {
    'chemical_exposure': """🚨 **화학물질 노출 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 즉시 행동**
- 오염 부위를 흐르는 깨끗한 물로 **최소 15~20분** 이상 세척하세요
- 눈에 들어간 경우: 눈을 뜬 채로 흐르는 물로 세척 (콘택트렌즈 먼저 제거)
- 옷에 묻은 경우: 오염된 옷을 즉시 벗기세요
- ⚡ **불산(HF) 노출 시**: 세척 후 **즉시 병원 이동** — 불산은 피부를 투과하여 전신 중독을 일으킬 수 있습니다

**2. 119 신고**
- 📞 **119**에 전화하세요
- 화학물질 이름과 양, 노출 부위를 알려주세요
- MSDS(물질안전보건자료)가 있으면 가져가세요

**3. 하지 마세요 ❌**
- 토하게 하지 마세요 (삼킨 경우)
- 중화제를 사용하지 마세요
- 오염 부위를 문지르지 마세요

**4. 병원 갈 때**
- 화학물질 이름/MSDS를 꼭 가져가세요
- 오염된 옷을 비닐에 담아 가져가세요

⚠️ **현장실습생은 반드시 담당 교사와 현장 관리자에게도 알리세요!**""",

    'injury_bleeding': """🚨 **외상/출혈 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 지혈**
- 깨끗한 천이나 거즈로 상처 부위를 **강하게 눌러주세요**
- 심장보다 높은 위치로 올리세요
- 10분 이상 꾹 누르고 있으세요 (중간에 열어보지 마세요)

**2. 119 신고가 필요한 경우**
- 📞 **119** — 피가 안 멈추거나, 뼈가 보이거나, 손가락/발가락 절단 시
- 절단된 부위: 깨끗한 비닐에 넣고 얼음물에 담아 병원에 가져가세요

**3. 하지 마세요 ❌**
- 상처에 이물질 제거를 무리하게 하지 마세요
- 부러진 뼈를 억지로 맞추지 마세요
- 탈구된 관절을 억지로 끼우지 마세요

**4. 현장실습 중이라면**
- 담당 교사와 현장 관리자에게 즉시 알리세요
- 산재보험 적용 대상입니다 — 치료비 걱정하지 마세요

⚠️ **기억하세요: 여러분의 안전이 가장 중요합니다!**""",

    'burn': """🚨 **화상 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 즉시 냉각**
- 흐르는 **시원한 물**(차갑지 않은)로 **20분 이상** 식히세요
- 얼음이나 얼음물은 사용하지 마세요 (동상 위험)

**2. 화상 부위 보호**
- 물집은 터뜨리지 마세요
- 깨끗한 거즈나 랩으로 가볍게 덮으세요
- 옷이 달라붙었으면 억지로 떼지 마세요

**3. 119 신고가 필요한 경우**
- 📞 **119** — 화상 범위가 손바닥보다 크거나, 얼굴/손/관절 화상, 3도 화상(피부 하얗거나 까맣게 탄)

**4. 하지 마세요 ❌**
- 된장, 치약, 소주 등 민간요법 ❌
- 물집 터뜨리기 ❌
- 얼음 직접 대기 ❌

⚠️ **현장실습생은 반드시 담당 교사와 현장 관리자에게도 알리세요!**""",

    'electric_shock': """🚨 **감전 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 안전 확보 (본인 안전 우선!)**
- ⚡ **전원을 먼저 차단**하세요 (차단기/스위치)
- 전원 차단이 안 되면: 마른 나무막대, 고무장갑 등 절연체로 감전자를 전원에서 분리
- **맨손으로 절대 만지지 마세요** — 같이 감전됩니다!

**2. 119 신고**
- 📞 **119** 즉시 신고
- 감전 사고는 외부에 안 보여도 내부 손상이 클 수 있습니다
- 반드시 병원 검사를 받아야 합니다

**3. 응급처치**
- 의식이 없으면: 기도 확보 → 호흡 확인 → CPR (심폐소생술)
- 화상 부위가 있으면: 깨끗한 천으로 덮기

**4. 하지 마세요 ❌**
- 전원 차단 전에 감전자 만지기 ❌
- 물을 뿌리기 ❌

⚠️ **감전은 가벼워 보여도 반드시 병원 검사를 받으세요!**""",

    'breathing_difficulty': """🚨 **호흡곤란/의식불명 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 즉시 행동**
- 환자를 유해 환경(가스, 분진, 밀폐공간)에서 **즉시 이동**시키세요
- 신선한 공기가 있는 곳으로 옮기세요
- 옷의 목 부분, 벨트를 풀어 호흡을 편하게 해주세요

**2. 119 신고**
- 📞 **119** 즉시 신고
- 증상과 원인(가스 흡입, 분진 등)을 알려주세요

**3. 의식이 없는 경우**
- 고개를 뒤로 젖혀 기도를 확보하세요
- 호흡이 없으면: CPR (심폐소생술) 시작
  - 가슴 중앙을 **분당 100~120회** 속도로 압박
  - AED(자동제세동기)가 있으면 즉시 사용

**4. 하지 마세요 ❌**
- 밀폐공간에 구조하러 들어가지 마세요 (2차 사고 위험!)
- 의식 없는 사람에게 물을 먹이지 마세요

⚠️ **밀폐공간 사고는 혼자 구조하면 안 됩니다! 반드시 119를 먼저 부르세요!**""",

    'fall_crush': """🚨 **추락/협착/붕괴 — 응급조치 매뉴얼**

> ⚠️ 이 정보는 응급 참고용이며, 전문 응급처치 교육을 대체하지 않습니다.

**1. 안전 확보**
- 추가 붕괴/낙하 위험이 없는지 먼저 확인하세요
- 안전한 경우에만 접근하세요

**2. 119 신고**
- 📞 **119** 즉시 신고
- 사고 유형, 부상자 수, 위치를 알려주세요

**3. 부상자 보호**
- 목/척추 부상이 의심되면 **절대 움직이지 마세요**
- 출혈이 있으면 깨끗한 천으로 눌러 지혈
- 체온 유지를 위해 담요 등으로 덮어주세요

**4. 하지 마세요 ❌**
- 목/허리를 다친 것 같으면 절대 움직이지 마세요
- 무거운 물체에 깔린 경우 함부로 들어올리지 마세요 (크러시 증후군 위험)
- 2차 사고 위험이 있으면 무리하게 구조하지 마세요

⚠️ **추락/협착 사고는 산재보험 적용 대상입니다! 반드시 신고하세요!**""",
}

_EMERGENCY_FOOTER = """
---
📌 **현장실습생 긴급 연락처**
- 🚑 **119** (소방/응급)
- ☎️ **1350** (고용노동부 — 산재/노동 상담)
- ☎️ **1644-8585** (안전보건공단)
- 담당 교사, 현장 관리자에게도 반드시 알리세요

💡 이 응답은 응급조치 가이드입니다. 정확한 진단과 치료는 의료 전문가에게 받으세요.
"""


def get_emergency_response(category: str) -> str:
    """Return pre-written emergency manual for the given category.

    Falls back to injury_bleeding if category is unknown.
    """
    manual = EMERGENCY_MANUALS.get(category, EMERGENCY_MANUALS['injury_bleeding'])
    return manual + _EMERGENCY_FOOTER
