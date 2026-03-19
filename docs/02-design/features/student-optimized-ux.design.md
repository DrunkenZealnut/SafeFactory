# Design: 직업계고 학생 맞춤형 시스템 최적화

> Plan 문서: `docs/01-plan/features/student-optimized-ux.plan.md`

---

## 1. 아키텍처 개요

### 1.1 변경 흐름도

```
사용자 질문 입력
       │
       ▼
┌──────────────────────────┐
│ Pre-Phase: Emergency     │ ← [Sub-Feature A] 신규
│ Intent Classifier        │
│ (services/emergency_     │
│  responder.py)           │
└──────────┬───────────────┘
           │ 응급 아님
           ▼
┌──────────────────────────┐
│ Pre-Phase: Semantic      │ ← [Sub-Feature C] 신규
│ Cache Lookup             │
│ (services/semantic_      │
│  cache.py)               │
└──────────┬───────────────┘
           │ 캐시 미스
           ▼
┌──────────────────────────┐
│ Phase 0: Domain          │   기존 유지
│ Classification           │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Phase 1: Query           │ ← [Sub-Feature B] 확장
│ Enhancement              │   YOUTH_COLLOQUIAL_MAP 추가
│ + 학생 구어체 번역       │   Multi-Query 프롬프트 강화
└──────────┬───────────────┘
           ▼
   Phase 2~6 (기존 유지)
           │
           ▼
┌──────────────────────────┐
│ Phase 7: Context Build   │ ← [Sub-Feature D] 확장
│ + 이미지 메타데이터      │   이미지 정보를 LLM 컨텍스트에 주입
│   LLM 컨텍스트 주입      │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Phase 8: LLM Generation  │ ← [Sub-Feature D] 확장
│ + 학생 눈높이 프롬프트   │   DOMAIN_PROMPTS 전체 수정
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Post-Phase: Semantic     │ ← [Sub-Feature C] 신규
│ Cache Store              │   성공 응답 캐시 저장
└──────────────────────────┘
```

### 1.2 변경 파일 목록

| 파일 | 변경 유형 | Sub-Feature | 영향도 |
|------|----------|-------------|--------|
| `services/emergency_responder.py` | **신규** | A | - |
| `services/semantic_cache.py` | **신규** | C | - |
| `services/rag_pipeline.py` | 수정 | A, C, D | 높음 |
| `services/query_router.py` | 수정 | A | 중간 |
| `src/query_enhancer.py` | 수정 | B | 중간 |
| `services/domain_config.py` | 수정 | D | 낮음 |
| `api/v1/search.py` | 수정 | A, C | 중간 |
| `templates/domain.html` | 수정 | D | 낮음 |

---

## 2. Sub-Feature A: 응급/긴급 상황 Fast-Track

### 2.1 모듈 설계: `services/emergency_responder.py`

#### 2.1.1 EmergencyClassifier 클래스

```python
# services/emergency_responder.py

import re
import logging
from typing import Optional, Tuple

class EmergencyClassifier:
    """응급 의도 분류기 — 키워드 기반 1차 분류 + 카테고리 매핑."""

    # 응급 카테고리별 키워드 사전
    EMERGENCY_PATTERNS = {
        'chemical_exposure': {
            'keywords': [
                '화학물질', '약품', '쏟았', '튀었', '눈에 들어', '피부에 묻',
                '삼켰', '마셨', '흡입', '냄새가 심', '가스 누출', '누출',
                '불산', '염산', '황산', '유독', '중독',
            ],
            'patterns': [
                re.compile(r'(?:화학|약품|산|알칼리|용제).*(?:쏟|튀|묻|닿|들어갔|접촉)'),
                re.compile(r'(?:눈|피부|손|얼굴).*(?:화학|약품|산|따가|아프|빨갛)'),
            ],
        },
        'injury_bleeding': {
            'keywords': [
                '다쳤', '다쳐', '베였', '베어', '찢어', '피가 나', '피가 안 멈',
                '출혈', '상처', '부러', '골절', '끼었', '끼임', '절단',
                '찍혔', '찔렸', '맞았',
            ],
            'patterns': [
                re.compile(r'(?:손|팔|다리|발|머리|눈).*(?:다치|베이|찢|부러|끼)'),
                re.compile(r'(?:피|출혈).*(?:나|안 멈|많|심)'),
            ],
        },
        'burn': {
            'keywords': [
                '화상', '데었', '데여', '뜨거운', '불꽃', '스파크', '용접',
                '증기', '열탕', '화염',
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
                '추락', '떨어졌', '무거운 것', '깔렸', '매달려', '끼었',
                '협착', '무너졌', '붕괴',
            ],
            'patterns': [
                re.compile(r'(?:높은 곳|사다리|지붕|비계).*(?:떨어|추락|미끄러)'),
            ],
        },
    }

    @classmethod
    def classify(cls, query: str) -> Optional[Tuple[str, float]]:
        """응급 의도 분류.

        Args:
            query: 사용자 질문 원문.

        Returns:
            (category, confidence) 튜플. 응급이 아니면 None.
            confidence: 0.0~1.0 (키워드 매치 수 기반)
        """
        query_lower = query.lower()
        best_category = None
        best_score = 0

        for category, config in cls.EMERGENCY_PATTERNS.items():
            score = 0
            # 키워드 매치
            for kw in config['keywords']:
                if kw in query_lower:
                    score += 1
            # 패턴 매치 (가중치 2)
            for pattern in config['patterns']:
                if pattern.search(query_lower):
                    score += 2

            if score > best_score:
                best_score = score
                best_category = category

        if best_score >= 1:
            confidence = min(best_score / 5.0, 1.0)
            logging.info(
                "[Emergency] Detected: %s (score=%d, conf=%.2f) for '%.40s'",
                best_category, best_score, confidence, query
            )
            return best_category, confidence

        return None
```

#### 2.1.2 응급조치 매뉴얼 사전

```python
# services/emergency_responder.py (계속)

EMERGENCY_MANUALS = {
    'chemical_exposure': """🚨 **화학물질 노출 — 응급조치 매뉴얼**

**1. 즉시 행동**
- 오염 부위를 흐르는 깨끗한 물로 **최소 15~20분** 이상 세척하세요
- 눈에 들어간 경우: 눈을 뜬 채로 흐르는 물로 세척 (콘택트렌즈 먼저 제거)
- 옷에 묻은 경우: 오염된 옷을 즉시 벗기세요

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

# 응급 응답에 공통으로 붙는 꼬리말
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
    """카테고리별 응급조치 매뉴얼 반환."""
    manual = EMERGENCY_MANUALS.get(category, EMERGENCY_MANUALS['injury_bleeding'])
    return manual + _EMERGENCY_FOOTER
```

### 2.2 통합 지점

#### 2.2.1 `api/v1/search.py` — `/ask` 엔드포인트

```python
# api/v1/search.py — api_ask() 함수 최상단에 추가

from services.emergency_responder import EmergencyClassifier, get_emergency_response

@v1_bp.route('/ask', methods=['POST'])
@rate_limit("20 per minute")
def api_ask():
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        query = data.get('query', '').strip()

        # === Pre-Phase: Emergency Fast-Track ===
        emergency = EmergencyClassifier.classify(query)
        if emergency:
            category, confidence = emergency
            answer = get_emergency_response(category)
            logging.info("[Emergency] Fast-track response: %s (conf=%.2f)", category, confidence)
            return success_response(data={
                'query': query,
                'answer': answer,
                'sources': [],
                'source_count': 0,
                'images': [],
                'emergency': True,
                'emergency_category': category,
                'emergency_confidence': confidence,
            })

        # 기존 pipeline 호출 계속...
        pipeline = run_rag_pipeline(data)
        ...
```

#### 2.2.2 `api/v1/search.py` — `/ask/stream` 엔드포인트

```python
# api/v1/search.py — api_ask_stream() 함수 최상단에 동일 로직 추가

@v1_bp.route('/ask/stream', methods=['POST'])
@rate_limit("20 per minute")
def api_ask_stream():
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        query = data.get('query', '').strip()

        # === Pre-Phase: Emergency Fast-Track (SSE) ===
        emergency = EmergencyClassifier.classify(query)
        if emergency:
            category, confidence = emergency
            answer = get_emergency_response(category)

            def emergency_generate():
                # 메타데이터 이벤트
                meta = json.dumps({
                    'type': 'metadata',
                    'data': {
                        'query': query, 'sources': [], 'source_count': 0,
                        'images': [], 'emergency': True,
                        'emergency_category': category,
                    }
                }, ensure_ascii=False)
                yield f"data: {meta}\n\n"

                # 응답 전체를 한 번에 전송 (즉시 표시)
                chunk = json.dumps({
                    'type': 'content', 'data': answer
                }, ensure_ascii=False)
                yield f"data: {chunk}\n\n"

                done = json.dumps({'type': 'done', 'data': {
                    'confidence': confidence,
                }}, ensure_ascii=False)
                yield f"data: {done}\n\n"

            return Response(
                stream_with_context(emergency_generate()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
            )

        # 기존 pipeline 호출 계속...
        pipeline = run_rag_pipeline(data)
        ...
```

#### 2.2.3 `services/query_router.py` — emergency 타입 추가

```python
# query_router.py — QUERY_TYPE_CONFIG에 추가

QUERY_TYPE_CONFIG = {
    'emergency': {
        'top_k_mult': 1,
        'use_hyde': False,
        'use_multi_query': False,
        'rerank_weight': 1.0,
        'skip_pipeline': True,  # 파이프라인 우회 플래그
    },
    'factual': { ... },  # 기존 유지
    ...
}
```

### 2.3 프론트엔드 — 응급 응답 시각적 구분

```javascript
// templates/domain.html — renderAnswer() 함수 내 분기 추가

if (data.emergency) {
    // 응급 응답 스타일 적용
    answerEl.classList.add('emergency-response');
    // 응급 배너 표시
    const banner = document.createElement('div');
    banner.className = 'emergency-banner';
    banner.innerHTML = '🚨 응급조치 가이드 — 즉시 119에 신고하세요!';
    answerEl.parentElement.prepend(banner);
}
```

```css
/* static/css/theme.css 추가 */
.emergency-response {
    border-left: 4px solid #ef4444;
    background: linear-gradient(135deg, #fef2f2 0%, #fff 100%);
    padding: 1.5rem;
    border-radius: 8px;
}
.emergency-banner {
    background: #ef4444;
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}
```

---

## 3. Sub-Feature B: 학생 눈높이 쿼리 번역기

### 3.1 `src/query_enhancer.py` 변경

#### 3.1.1 YOUTH_COLLOQUIAL_MAP 추가 (L184 이후)

```python
# src/query_enhancer.py — DOMAIN_SYNONYMS 바로 뒤에 추가

# ---------------------------------------------------------------------------
# 청소년/학생 구어체 → 산업/노동법 전문 용어 매핑
# ---------------------------------------------------------------------------
YOUTH_COLLOQUIAL_MAP = {
    # === 노동 관련 ===
    '알바': ['아르바이트', '단시간근로', '시간제근로'],
    '알바하다': ['근로', '시간제근로'],
    '쌤': ['담당교사', '현장실습교사'],
    '사장님': ['사업주', '사용자'],
    '월급': ['임금', '급여'],
    '월급 안 줘': ['임금체불', '체불임금', '근로기준법 위반'],
    '돈 안 줘': ['임금체불', '급여 미지급'],
    '야근': ['연장근로', '초과근무'],
    '야근 강요': ['연장근로 강제', '청소년 근로시간 제한'],
    '잘리다': ['해고', '부당해고'],
    '짤리다': ['해고', '부당해고'],
    '퇴직금 안 줘': ['퇴직급여 미지급'],
    '쉬는 날': ['휴일', '휴무일', '주휴일'],
    '계약서': ['근로계약서', '표준근로계약서'],
    '계약서 안 써': ['근로계약서 미작성'],

    # === 부상/건강 관련 ===
    '다쳤어': ['부상', '산업재해', '산재'],
    '아파': ['건강 이상', '직업병'],
    '베였어': ['절상', '외상', '부상'],
    '허리 아파': ['근골격계 질환', '요통'],
    '손목 아파': ['근골격계 질환', '수근관 증후군'],
    '피부 트러블': ['직업성 피부질환', '접촉성 피부염'],
    '귀가 안 들려': ['소음성 난청', '청력 손실'],
    '눈이 아파': ['안구 손상', '시력 장해'],

    # === 화학물질 관련 ===
    '약품': ['화학물질', '유해물질'],
    '약품 냄새': ['유해가스 노출', '화학물질 흡입'],
    '냄새가 심해': ['유해가스', '환기 부족', '허용농도 초과'],
    '세척액': ['세정제', '유기용제'],
    '가루': ['분진', '미세먼지', '입자상 물질'],

    # === 작업환경 관련 ===
    '시끄러워': ['소음', '청력보존프로그램'],
    '더워': ['고온작업', '열사병 예방'],
    '추워': ['한냉작업', '저체온증'],
    '어두워': ['조도 부족', '작업장 조명'],
    '무거워': ['중량물 취급', '인력운반 작업'],
    '높은 데': ['고소작업', '추락 위험'],
    '좁은 데': ['밀폐공간', '밀폐공간 작업'],

    # === 보호구 관련 ===
    '마스크': ['방진마스크', '호흡보호구', '방독마스크'],
    '장갑': ['보호장갑', '내화학장갑', '절연장갑'],
    '안경': ['보안경', '보호안경'],
    '모자': ['안전모', '보호모'],
    '귀마개': ['귀마개', '귀덮개', '청력보호구'],

    # === 권리/제도 관련 ===
    '보험': ['4대보험', '산재보험', '고용보험'],
    '보건증': ['건강진단결과서', '채용시건강진단'],
    '성희롱': ['직장 내 성희롱', '성희롱 예방교육'],
    '괴롭힘': ['직장 내 괴롭힘', '근로기준법 제76조의2'],
    '폭언': ['직장 내 괴롭힘', '언어폭력'],
}
```

#### 3.1.2 expand_with_synonyms() 확장

```python
# src/query_enhancer.py — expand_with_synonyms 메서드 수정

def expand_with_synonyms(self, query: str, domain: str = '') -> str:
    """Expand query with domain-specific synonyms AND youth colloquial mapping."""
    added = []
    query_lower = query.lower()

    # 1. 기존 도메인 동의어 확장 (유지)
    synonyms_map = DOMAIN_SYNONYMS.get(domain, {})
    for term, syns in synonyms_map.items():
        if term.lower() in query_lower:
            for syn in syns:
                if syn.lower() not in query_lower and syn not in added:
                    added.append(syn)
                    if len(added) >= 5:
                        break
        if len(added) >= 5:
            break

    # 2. 청소년 구어체 → 전문 용어 매핑 (신규)
    youth_added = []
    for colloquial, formal_terms in YOUTH_COLLOQUIAL_MAP.items():
        if colloquial in query_lower:
            for term in formal_terms:
                if term.lower() not in query_lower and term not in added and term not in youth_added:
                    youth_added.append(term)
                    if len(youth_added) >= 4:  # 구어체 매핑은 4개까지
                        break
        if len(youth_added) >= 4:
            break

    all_added = added + youth_added
    if all_added:
        return f"{query} ({' '.join(all_added)})"
    return query
```

#### 3.1.3 Multi-Query 프롬프트 강화

```python
# src/query_enhancer.py — generate_multi_queries() 내 시스템 프롬프트 수정
# 기존 프롬프트 뒤에 학생 맥락 추가

_MULTI_QUERY_SYSTEM_PROMPT_SUFFIX = """

## 추가 고려사항 (직업계고 학생 대상)
- 질문자는 산업 현장 경험이 부족한 직업계고 학생(청소년)일 수 있습니다
- 구어체나 일상적 표현이 있으면, 그에 대응하는 산업/노동법 전문 용어로도 검색 쿼리를 생성하세요
- 질문의 이면에 있는 '법적 보호 필요성'이나 '잠재적 위험 요인'을 추론하여 쿼리를 확장하세요
  - 예: "알바하다 손 다쳤어" → "산업재해 치료", "산재보험 신청", "청소년 근로자 보호"
  - 예: "약품 냄새가 너무 심해" → "유해화학물질 노출 기준", "작업환경측정", "환기 설비"
"""
```

### 3.2 법적 보호 키워드 부스팅

```python
# src/query_enhancer.py — enhance_query() 메서드 내 추가 로직

# 청소년 근로 관련 키워드 감지 → 보호 키워드 자동 추가
_YOUTH_LABOR_TRIGGERS = ['알바', '현장실습', '인턴', '실습생', '청소년', '학생', '고등학']
_YOUTH_PROTECTION_BOOST = ['청소년근로기준법', '근로계약서 필수', '18세 미만 근로 제한']

def _detect_youth_context(query: str) -> bool:
    """청소년 근로 맥락 감지."""
    return any(kw in query for kw in _YOUTH_LABOR_TRIGGERS)
```

---

## 4. Sub-Feature C: 시맨틱 캐싱 계층

### 4.1 모듈 설계: `services/semantic_cache.py`

```python
# services/semantic_cache.py

import hashlib
import json
import logging
import numpy as np
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

class SemanticCache:
    """의미 기반 캐시 — 유사 질문의 이전 답변을 즉시 반환."""

    DB_PATH = Path(__file__).parent.parent / 'instance' / 'semantic_cache.db'
    DEFAULT_TTL = 3600       # 1시간 (일반 질문)
    FAQ_TTL = 86400          # 24시간 (FAQ성 질문)
    MAX_ENTRIES = 1000       # 최대 캐시 항목
    SIMILARITY_THRESHOLD = 0.95  # 코사인 유사도 임계값

    def __init__(self):
        self._lock = threading.Lock()
        self._init_db()
        # 메모리 캐시: {namespace: [(embedding_array, cache_key), ...]}
        self._embedding_index: Dict[str, list] = {}
        self._load_embeddings()

    def _init_db(self):
        """SQLite 테이블 초기화."""
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_key TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                namespace TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl INTEGER NOT NULL,
                hit_count INTEGER DEFAULT 0,
                is_faq BOOLEAN DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_namespace ON semantic_cache(namespace)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_created ON semantic_cache(created_at)")
        conn.commit()
        conn.close()

    def _load_embeddings(self):
        """DB에서 임베딩을 메모리로 로드 (빠른 유사도 검색용)."""
        conn = sqlite3.connect(str(self.DB_PATH))
        rows = conn.execute(
            "SELECT cache_key, namespace, query_embedding FROM semantic_cache"
        ).fetchall()
        conn.close()

        index = {}
        for key, ns, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            index.setdefault(ns, []).append((emb, key))
        self._embedding_index = index

    def lookup(self, query_embedding: np.ndarray, namespace: str) -> Optional[Dict[str, Any]]:
        """캐시 조회 — 유사도 > SIMILARITY_THRESHOLD이면 히트."""
        ns_entries = self._embedding_index.get(namespace, [])
        if not ns_entries:
            return None

        best_sim = 0.0
        best_key = None
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for emb, key in ns_entries:
            e_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(q_norm, e_norm))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self.SIMILARITY_THRESHOLD and best_key:
            # TTL 확인 및 응답 반환
            conn = sqlite3.connect(str(self.DB_PATH))
            row = conn.execute(
                "SELECT response_json, created_at, ttl FROM semantic_cache WHERE cache_key = ?",
                (best_key,)
            ).fetchone()

            if row:
                resp_json, created_at, ttl = row
                if time.time() - created_at < ttl:
                    conn.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                        (best_key,)
                    )
                    conn.commit()
                    conn.close()
                    logging.info("[SemanticCache] HIT (sim=%.4f) for ns=%s", best_sim, namespace)
                    return json.loads(resp_json)
                else:
                    # TTL 만료 → 삭제
                    conn.execute("DELETE FROM semantic_cache WHERE cache_key = ?", (best_key,))
                    conn.commit()
            conn.close()

        return None

    def store(self, query_text: str, query_embedding: np.ndarray,
              namespace: str, response: Dict[str, Any], is_faq: bool = False):
        """캐시 저장."""
        with self._lock:
            self._enforce_size_limit()
            cache_key = hashlib.md5(
                f"{query_text}:{namespace}".encode()
            ).hexdigest()
            ttl = self.FAQ_TTL if is_faq else self.DEFAULT_TTL
            emb_blob = query_embedding.astype(np.float32).tobytes()

            conn = sqlite3.connect(str(self.DB_PATH))
            conn.execute("""
                INSERT OR REPLACE INTO semantic_cache
                (cache_key, query_text, query_embedding, namespace, response_json, created_at, ttl, is_faq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, query_text, emb_blob, namespace,
                  json.dumps(response, ensure_ascii=False), time.time(), ttl, is_faq))
            conn.commit()
            conn.close()

            # 메모리 인덱스 업데이트
            self._embedding_index.setdefault(namespace, []).append(
                (query_embedding.astype(np.float32), cache_key)
            )

    def invalidate_namespace(self, namespace: str):
        """특정 네임스페이스 캐시 전체 삭제."""
        conn = sqlite3.connect(str(self.DB_PATH))
        conn.execute("DELETE FROM semantic_cache WHERE namespace = ?", (namespace,))
        conn.commit()
        conn.close()
        self._embedding_index.pop(namespace, None)
        logging.info("[SemanticCache] Invalidated namespace: %s", namespace)

    def _enforce_size_limit(self):
        """캐시 크기 제한 — LRU 방식으로 오래된 항목 삭제."""
        conn = sqlite3.connect(str(self.DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        if count >= self.MAX_ENTRIES:
            # 가장 오래되고 히트 수 적은 20% 삭제
            delete_count = self.MAX_ENTRIES // 5
            conn.execute("""
                DELETE FROM semantic_cache WHERE cache_key IN (
                    SELECT cache_key FROM semantic_cache
                    ORDER BY hit_count ASC, created_at ASC
                    LIMIT ?
                )
            """, (delete_count,))
            conn.commit()
            self._load_embeddings()  # 메모리 인덱스 재구성
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환 (관리자 패널용)."""
        conn = sqlite3.connect(str(self.DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        total_hits = conn.execute("SELECT SUM(hit_count) FROM semantic_cache").fetchone()[0] or 0
        faq_count = conn.execute("SELECT COUNT(*) FROM semantic_cache WHERE is_faq = 1").fetchone()[0]
        by_ns = conn.execute(
            "SELECT namespace, COUNT(*), SUM(hit_count) FROM semantic_cache GROUP BY namespace"
        ).fetchall()
        conn.close()
        return {
            'total_entries': total,
            'total_hits': total_hits,
            'faq_entries': faq_count,
            'max_entries': self.MAX_ENTRIES,
            'by_namespace': {ns: {'entries': c, 'hits': h} for ns, c, h in by_ns},
        }
```

### 4.2 통합 지점

#### 4.2.1 `services/singletons.py` — 싱글턴 추가

```python
# services/singletons.py

_semantic_cache = None
_semantic_cache_lock = threading.RLock()

def get_semantic_cache():
    global _semantic_cache
    if _semantic_cache is None:
        with _semantic_cache_lock:
            if _semantic_cache is None:
                from services.semantic_cache import SemanticCache
                _semantic_cache = SemanticCache()
    return _semantic_cache

def invalidate_semantic_cache(namespace: str = None):
    global _semantic_cache
    if _semantic_cache and namespace:
        _semantic_cache.invalidate_namespace(namespace)
```

#### 4.2.2 `api/v1/search.py` — 캐시 조회/저장

```python
# api/v1/search.py — api_ask() 내 Emergency 체크 이후, pipeline 호출 이전

# === Pre-Phase: Semantic Cache Lookup ===
from services.singletons import get_semantic_cache, get_agent
namespace_hint = data.get('namespace', '')
cache = get_semantic_cache()

# 캐시 조회를 위한 쿼리 임베딩 생성
agent = get_agent()
query_embedding = agent.generate_embedding(query)  # np.ndarray 반환
cached = cache.lookup(query_embedding, namespace_hint)
if cached:
    logging.info("[SemanticCache] Returning cached response for: %.40s", query)
    cached['cache_hit'] = True
    return success_response(data=cached)

# ... (기존 pipeline 호출) ...

# === Post-Phase: 성공 응답 캐시 저장 ===
# api_ask() 응답 반환 직전
if resp_data.get('answer') and len(resp_data.get('sources', [])) > 0:
    try:
        cache.store(
            query_text=query,
            query_embedding=query_embedding,
            namespace=namespace,
            response=resp_data,
        )
    except Exception as e:
        logging.warning("[SemanticCache] Store failed: %s", e)
```

### 4.3 캐시 관리 API (관리자)

```python
# api/v1/admin.py — 추가 엔드포인트

@v1_bp.route('/admin/cache/stats', methods=['GET'])
def admin_cache_stats():
    """시맨틱 캐시 통계."""
    cache = get_semantic_cache()
    return success_response(data=cache.get_stats())

@v1_bp.route('/admin/cache/invalidate', methods=['POST'])
def admin_cache_invalidate():
    """특정 네임스페이스 캐시 무효화."""
    ns = request.json.get('namespace')
    if ns:
        invalidate_semantic_cache(ns)
    return success_response(data={'invalidated': ns})
```

---

## 5. Sub-Feature D: 답변 난이도 조절 & 시각적 맥락 강화

### 5.1 `services/domain_config.py` — DOMAIN_PROMPTS 수정

모든 도메인 프롬프트에 아래 **학생 눈높이 지시문 블록**을 공통 추가:

```python
# services/domain_config.py — 공통 학생 눈높이 지시문

STUDENT_FRIENDLY_INSTRUCTIONS = """

## 답변 어투 및 난이도 지침 (직업계고 학생 대상)
- **쉬운 어휘**: 고등학교 1~2학년 수준의 쉬운 단어를 사용하세요
- **친근한 어투**: ~요/~다 체를 사용하고, 딱딱한 문어체를 피하세요
- **전문 용어 풀어쓰기**: 전문 용어를 사용할 때는 반드시 괄호 안에 쉬운 설명을 병기하세요
  - 예: "개인보호구(PPE, 작업할 때 안전을 위해 착용하는 장비)"
  - 예: "위험성평가(작업장에서 어떤 위험이 있는지 미리 조사하는 것)"
- **핵심 먼저**: 가장 중요한 내용을 먼저 말하고, 자세한 설명은 뒤에 붙이세요
- **단계별 구조화**: 절차나 방법을 설명할 때 반드시 번호를 매겨서 단계별로 정리하세요
- **짧은 문장**: 한 문장이 너무 길지 않게 끊어 쓰세요 (40자 이내 권장)
- **공감 표현**: 학생이 겪는 상황에 공감하는 표현을 적절히 사용하세요
  - 예: "처음이라 당연히 궁금할 수 있어요", "걱정되는 마음 이해해요"
"""
```

#### 적용 방법

```python
# services/domain_config.py — build_llm_prompts() 내부 또는 DOMAIN_PROMPTS 조합 시

# 기존:
# system_prompt = base_prompt + COT_INSTRUCTIONS + domain_cot + VISUAL_GUIDELINES

# 변경:
system_prompt = base_prompt + STUDENT_FRIENDLY_INSTRUCTIONS + COT_INSTRUCTIONS + domain_cot + VISUAL_GUIDELINES
```

### 5.2 이미지 메타데이터 LLM 컨텍스트 주입

#### 5.2.1 `services/rag_pipeline.py` — Phase 7 확장

```python
# services/rag_pipeline.py — build_llm_prompts() 수정

def build_llm_prompts(query, sources, context, namespace, ...,
                      related_images=None):  # 새 파라미터 추가
    """..."""
    # 기존 프롬프트 조합
    system_prompt = base_prompt + STUDENT_FRIENDLY_INSTRUCTIONS + COT_INSTRUCTIONS + domain_cot + VISUAL_GUIDELINES

    # 이미지 메타데이터를 user_prompt에 추가
    if related_images:
        image_context = "\n## 관련 이미지 (시스템이 자동 표시)\n"
        for i, img in enumerate(related_images[:5], 1):  # 최대 5개
            image_context += f"- [이미지 {i}] {img['name']}\n"
        image_context += "\n**참고**: 위 이미지가 사용자 화면에 함께 표시됩니다. 답변에서 관련 이미지를 자연스럽게 언급해주세요 (예: '아래 이미지에서 확인할 수 있어요').\n"
        user_prompt += image_context

    return system_prompt, user_prompt
```

#### 5.2.2 `api/v1/search.py` — 이미지 정보 전달

```python
# api/v1/search.py — api_ask() 내

# 기존: messages = build_llm_messages(query, sources, context, namespace, ...)
# 변경: 이미지 수집 후 build_llm_messages에 전달

related_images = _collect_related_images(sources)
messages = build_llm_messages(query, sources, context, namespace,
                              calc_result, law_refs_formatted,
                              labor_classification, legal_analysis,
                              safety_refs, msds_refs,
                              related_images=related_images)  # 추가
```

### 5.3 프론트엔드 — 인라인 이미지 참조

```javascript
// templates/domain.html — 답변 렌더링 후처리

function enhanceAnswerWithImages(answerHtml, images) {
    if (!images || images.length === 0) return answerHtml;

    // "아래 이미지", "이미지를 참고", "이미지에서 확인" 등의 패턴 감지
    const imgRefPattern = /(?:아래\s*)?이미지(?:를?\s*참고|에서\s*확인|를?\s*보|를?\s*확인)/g;
    let imgIndex = 0;

    return answerHtml.replace(imgRefPattern, (match) => {
        if (imgIndex < images.length) {
            const img = images[imgIndex++];
            return `${match} <span class="inline-image-ref" data-src="${img.path}" title="${img.name}">📸 ${img.name}</span>`;
        }
        return match;
    });
}
```

---

## 6. 구현 순서 체크리스트

### Phase 1: Sub-Feature A — 응급 Fast-Track (~3일)

- [ ] `services/emergency_responder.py` 신규 생성
  - [ ] `EmergencyClassifier` 클래스 (6개 카테고리)
  - [ ] `EMERGENCY_MANUALS` 정적 응답 (6개)
  - [ ] `get_emergency_response()` 함수
- [ ] `api/v1/search.py` 수정
  - [ ] `/ask` 엔드포인트 Pre-Phase 분기
  - [ ] `/ask/stream` 엔드포인트 Pre-Phase 분기
- [ ] `services/query_router.py` 수정
  - [ ] `QUERY_TYPE_CONFIG`에 `emergency` 타입 추가
- [ ] `templates/domain.html` 수정
  - [ ] 응급 응답 시각적 구분 (CSS + JS)
- [ ] `static/css/theme.css` 수정
  - [ ] `.emergency-response`, `.emergency-banner` 스타일

### Phase 2: Sub-Feature B — 쿼리 번역기 (~4일)

- [ ] `src/query_enhancer.py` 수정
  - [ ] `YOUTH_COLLOQUIAL_MAP` 사전 추가 (50+ 항목)
  - [ ] `expand_with_synonyms()` 확장 (구어체 매핑 통합)
  - [ ] Multi-Query 프롬프트에 학생 맥락 추가
  - [ ] `_detect_youth_context()` 함수 추가
  - [ ] 법적 보호 키워드 부스팅 로직

### Phase 3: Sub-Feature D — 답변 난이도 (~3일)

- [ ] `services/domain_config.py` 수정
  - [ ] `STUDENT_FRIENDLY_INSTRUCTIONS` 상수 추가
  - [ ] 모든 DOMAIN_PROMPTS에 학생 지시문 통합
- [ ] `services/rag_pipeline.py` 수정
  - [ ] `build_llm_prompts()` — 이미지 메타데이터 파라미터 추가
  - [ ] 이미지 정보 user_prompt 주입 로직
- [ ] `api/v1/search.py` 수정
  - [ ] `related_images`를 `build_llm_messages()`에 전달
- [ ] `templates/domain.html` 수정
  - [ ] `enhanceAnswerWithImages()` 함수 추가

### Phase 4: Sub-Feature C — 시맨틱 캐싱 (~5일)

- [ ] `services/semantic_cache.py` 신규 생성
  - [ ] `SemanticCache` 클래스 (SQLite 기반)
  - [ ] `lookup()` — 코사인 유사도 검색
  - [ ] `store()` — 캐시 저장
  - [ ] `invalidate_namespace()` — 네임스페이스별 무효화
  - [ ] `get_stats()` — 통계 조회
- [ ] `services/singletons.py` 수정
  - [ ] `get_semantic_cache()` 싱글턴 추가
  - [ ] `invalidate_semantic_cache()` 함수 추가
- [ ] `api/v1/search.py` 수정
  - [ ] `/ask` — 캐시 조회/저장 로직
  - [ ] `/ask/stream` — 캐시 히트 시 즉시 응답
- [ ] `api/v1/admin.py` 수정
  - [ ] `/admin/cache/stats` 엔드포인트
  - [ ] `/admin/cache/invalidate` 엔드포인트

---

## 7. 데이터 흐름 상세

### 7.1 응급 Fast-Track 데이터 흐름

```
POST /ask {"query": "손에 약품이 튀었어요"}
  │
  ├─ EmergencyClassifier.classify("손에 약품이 튀었어요")
  │   └─ keyword "약품" + "튀었" → category: "chemical_exposure", score: 3
  │
  ├─ get_emergency_response("chemical_exposure")
  │   └─ 정적 응급조치 매뉴얼 반환
  │
  └─ Response: {emergency: true, answer: "🚨 화학물질 노출...", sources: []}
     (전체 소요: < 50ms)
```

### 7.2 구어체 쿼리 번역 데이터 흐름

```
Query: "알바하다 베였어요 어떡해요"
  │
  ├─ YOUTH_COLLOQUIAL_MAP 매핑
  │   ├─ "알바" → ['아르바이트', '단시간근로', '시간제근로']
  │   └─ "베였어" → ['절상', '외상', '부상']
  │
  ├─ expand_with_synonyms() 결과
  │   └─ "알바하다 베였어요 어떡해요 (아르바이트 단시간근로 절상 산업재해)"
  │
  ├─ Multi-Query 생성 (학생 맥락 프롬프트)
  │   ├─ "아르바이트 중 절상 부상 응급처치"
  │   ├─ "단시간근로자 산업재해 산재보험 신청"
  │   └─ "청소년 근로자 부상 시 대처 방법"
  │
  └─ Vector Search: 3개 쿼리 × top_k → 풍부한 검색 결과
```

### 7.3 시맨틱 캐시 데이터 흐름

```
POST /ask {"query": "보건증 어떻게 발급받아?"}
  │
  ├─ Emergency check → 아님
  │
  ├─ Semantic Cache Lookup
  │   ├─ query_embedding = embed("보건증 어떻게 발급받아?")
  │   ├─ DB 검색: cosine_sim(query_emb, cached_embs)
  │   ├─ 최고 유사도: 0.97 ("보건증 발급 방법이요?")
  │   └─ 0.97 > 0.95 → CACHE HIT
  │
  └─ Response: cached_response (cache_hit: true)
     (전체 소요: < 200ms, 임베딩 생성 포함)
```

---

## 8. 에러 처리 및 폴백

| 상황 | 에러 처리 |
|------|----------|
| 응급 분류 실패 (예외) | 로깅 후 일반 파이프라인으로 폴백 |
| 구어체 매핑에서 오탐 | 매핑 결과는 검색 확장용이므로 영향 제한적 |
| 캐시 DB 접근 실패 | 로깅 후 캐시 우회, 일반 파이프라인 실행 |
| 캐시 히트 응답이 오래된 정보 | TTL로 자동 만료, 관리자 수동 무효화 가능 |
| 이미지 메타데이터 없음 | 이미지 컨텍스트 블록 생략 (기존 동작 유지) |
| 프롬프트 변경 후 품질 저하 | STUDENT_FRIENDLY_INSTRUCTIONS를 별도 상수로 분리 → 롤백 용이 |

---

## 9. 성능 예측

| 시나리오 | 현재 | 개선 후 | 비고 |
|----------|------|---------|------|
| 응급 질문 | 5~15초 | < 100ms | 파이프라인 완전 우회 |
| FAQ 캐시 히트 | 5~15초 | < 200ms | 임베딩 생성 ~100ms + 캐시 조회 ~50ms |
| 구어체 질문 | 검색 실패 | 정상 검색 | 매핑 오버헤드 < 1ms |
| 답변 생성 | 동일 | 동일 | 프롬프트 추가 ~200토큰 → LLM 비용 미미 |
| 이미지 컨텍스트 | 이미지와 답변 분리 | 답변에서 이미지 참조 | user_prompt 추가 ~100토큰 |
