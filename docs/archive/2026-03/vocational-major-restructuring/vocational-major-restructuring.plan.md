# 직업계고 전공별 학습 및 안전자료 제공 플랫폼 재구성

> **Summary**: SafeFactory를 토픽 기반 다중 도메인 시스템에서 직업계고 전공 중심 학습/안전자료 플랫폼으로 재구성
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-14
> **Status**: Draft

---

## Executive Summary

| 관점 | 내용 |
|------|------|
| **문제(Problem)** | 현재 SafeFactory는 반도체·노동법·안전보건 등 토픽 단위로 분리되어 있어, 직업계고 학생이 자기 전공에 맞는 학습자료와 안전자료를 통합적으로 이용하기 어려움 |
| **해결(Solution)** | 도메인 구조를 "전공(Major) 중심"으로 재구성하여, 각 전공별로 NCS 학습자료 + 안전보건자료 + 현장실습자료를 하나의 진입점에서 제공 |
| **기능/UX 효과(Function/UX Effect)** | 학생이 전공을 선택하면 해당 전공의 모든 학습·안전자료에 즉시 접근 가능. 전공 추가 시 설정 파일만 확장하면 되는 플러그인 구조 |
| **핵심 가치(Core Value)** | 직업계고 학생의 전공별 맞춤 학습 경험 제공 + 안전사고 예방을 위한 전공 특화 안전교육 통합 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory를 **직업계고(특성화고/마이스터고) 전공별 학습 및 안전자료 통합 플랫폼**으로 재구성한다. 현재 반도체 전공을 기반으로 구축된 시스템을 확장 가능한 아키텍처로 전환하여, 향후 전기·전자, 기계, 화공, 자동차, 건축 등 다양한 전공을 추가할 수 있도록 한다.

### 1.2 Background

**현재 상태(As-Is)**:
- SafeFactory는 5개 토픽 기반 도메인으로 운영: semiconductor-v2, laborlaw, field-training, kosha, msds
- 각 도메인은 독립적으로 존재하며, "반도체" 학생이 안전자료를 보려면 별도 도메인(kosha)으로 이동해야 함
- 도메인 간 관계성이 불명확 — 학생 입장에서 어떤 메뉴가 자기 전공과 관련 있는지 파악 어려움

**목표 상태(To-Be)**:
- **전공 중심 구조**: 학생이 "반도체과" 선택 → NCS 학습자료 + 반도체 안전자료 + 현장실습 가이드가 통합 제공
- **확장 가능**: 새로운 전공 추가 시 설정 파일 + 문서 업로드만으로 서비스 확장
- **공통 자료 공유**: 노동법, 일반안전(KOSHA 공통), MSDS 등은 전공 공통 자료로 모든 전공에서 접근 가능

**대상 사용자**:
- 직업계고(특성화고/마이스터고) 학생
- 산업체 현장실습 지도교사
- 학교 안전교육 담당교사

### 1.3 Related Documents

- 기존 PDCA 히스토리: 자동 네임스페이스 라우팅, KOSHA 안전 자동첨부, 공유질문 페이지 등
- `services/domain_config.py` — 현재 도메인 구성
- `services/query_router.py` — 현재 자동 라우팅 로직

---

## 2. Scope

### 2.1 In Scope

- [ ] **전공(Major) 중심 도메인 아키텍처 재설계** — domain_config.py 구조 전환
- [ ] **전공 선택 UI/UX** — 홈 화면에서 전공 선택 후 통합 학습환경 진입
- [ ] **전공별 네임스페이스 전략** — Pinecone 네임스페이스 재설계 (전공별 학습 + 안전 통합)
- [ ] **전공별 LLM 시스템 프롬프트** — 전공 특화 답변 생성
- [ ] **전공별 자동 라우팅** — query_router.py 전공 키워드 기반 확장
- [ ] **공통 자료(노동법, 일반안전, MSDS) 크로스 서치** — 전공 무관 공통 접근
- [ ] **전공 추가 플러그인 구조** — 설정만으로 새 전공 추가 가능한 구조
- [ ] **반도체 전공 마이그레이션** — 기존 semiconductor-v2 데이터를 새 구조로 매핑
- [ ] **문서 처리 파이프라인 전공 태깅** — CLI 문서 업로드 시 전공 메타데이터 자동 태깅

### 2.2 Out of Scope

- 새로운 전공(전기, 기계 등)의 실제 문서 수집 및 업로드 (인프라만 준비)
- 모바일 앱 개발
- 사용자 인증 체계 변경 (기존 OAuth 유지)
- Pinecone 인덱스 재생성 (기존 벡터 데이터 유지, 메타데이터 레이어에서 처리)
- 커뮤니티/포럼 기능 변경

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | 요구사항 | 우선순위 | 상태 |
|----|----------|----------|------|
| FR-01 | 전공 정의 설정 구조 — 전공별 이름, 아이콘, 색상, 키워드, 관련 네임스페이스 목록을 하나의 설정으로 관리 | High | Pending |
| FR-02 | 전공 선택 홈 화면 — 카드형 전공 목록에서 전공 선택 시 해당 전공 학습환경으로 진입 | High | Pending |
| FR-03 | 전공별 통합 검색 — 선택한 전공의 학습자료 + 안전자료 + 현장실습자료를 한 번에 검색 | High | Pending |
| FR-04 | 전공별 RAG 파이프라인 — 전공 특화 시스템 프롬프트로 답변 생성, 전공 관련 네임스페이스 다중 검색 | High | Pending |
| FR-05 | 공통 자료 크로스 서치 — 노동법, MSDS, 일반안전은 모든 전공에서 자동 접근 | Medium | Pending |
| FR-06 | 전공 추가 설정 인터페이스 — 새 전공 추가 시 domain_config.py에 전공 정의만 추가하면 자동 반영 | High | Pending |
| FR-07 | 전공별 샘플 질문 — 전공 진입 시 해당 전공 맞춤 예시 질문 표시 | Medium | Pending |
| FR-08 | 전공별 네비게이션 — 상단 네비게이션에서 전공 전환 가능 | Medium | Pending |
| FR-09 | 전공별 문서 업로드 태깅 — CLI에서 문서 처리 시 전공 메타데이터 자동 부여 | Medium | Pending |
| FR-10 | 기존 반도체 전공 완벽 호환 — 재구성 후에도 기존 반도체 관련 모든 기능 정상 작동 | High | Pending |

### 3.2 Non-Functional Requirements

| 카테고리 | 기준 | 측정 방법 |
|----------|------|-----------|
| 호환성 | 기존 Pinecone 벡터 데이터 재색인 없이 전환 | 전환 전후 검색 결과 비교 |
| 확장성 | 전공 추가 시 코드 변경 최소화 (설정 파일만 수정) | 새 전공 추가 시 변경 파일 수 ≤ 2개 |
| 성능 | 전공별 다중 네임스페이스 검색에서도 응답 3초 이내 | /ask 엔드포인트 응답 시간 |
| UX | 전공 선택 → 검색까지 2클릭 이내 | 사용자 테스트 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] 전공 설정 구조 구현 및 반도체 전공 마이그레이션 완료
- [ ] 전공 선택 홈 화면 동작
- [ ] 반도체 전공에서 학습 + 안전 통합 검색 정상 작동
- [ ] 공통 자료(노동법, MSDS) 크로스 서치 정상 작동
- [ ] 새 전공 추가 시 설정 파일만으로 동작 확인 (더미 전공으로 테스트)
- [ ] 기존 기능(커뮤니티, 공유질문, 북마크 등) 정상 작동

### 4.2 Quality Criteria

- [ ] 기존 반도체 검색 품질 저하 없음 (동일 쿼리 동일 품질)
- [ ] 전공 추가 플러그인 구조 검증 (더미 전공 "전기전자과" 추가 테스트)
- [ ] 에러 없이 모든 API 엔드포인트 정상 응답

---

## 5. Risks and Mitigation

| 리스크 | 영향 | 가능성 | 대응 방안 |
|--------|------|--------|-----------|
| Pinecone 다중 네임스페이스 검색 시 성능 저하 | High | Medium | 전공별 네임스페이스 병합 vs 다중쿼리 벤치마크 후 결정 |
| 기존 도메인 라우팅 로직 충돌 | Medium | Medium | 기존 라우팅을 전공 라우팅의 서브레이어로 유지 |
| 전공 추가 시 시스템 프롬프트 품질 편차 | Medium | High | 프롬프트 템플릿 + 전공별 변수 주입 방식으로 품질 일관성 유지 |
| 프론트엔드 대규모 리팩토링 필요 | High | Low | domain.html의 기존 도메인 분기 구조를 전공 분기로 자연스럽게 전환 |
| 반도체 외 전공 문서 부재 | Low | High | 아키텍처만 준비하고, 실제 문서는 단계적 추가 |

---

## 6. Architecture Considerations

### 6.1 Project Level

| Level | 특징 | 선택 |
|-------|------|:----:|
| **Starter** | 단순 구조 | ☐ |
| **Dynamic** | 기능별 모듈, BaaS 연동 | ☑ |
| **Enterprise** | 엄격한 레이어 분리, DI, 마이크로서비스 | ☐ |

> Dynamic 유지 — Flask 기반 모놀리식 구조에서 설정 기반 확장으로 충분

### 6.2 Key Architectural Decisions

| 결정 사항 | 선택지 | 선택 | 근거 |
|-----------|--------|------|------|
| 전공-네임스페이스 매핑 | 1:1 매핑 / 1:N 매핑 / 태그 기반 | **1:N 매핑** | 한 전공이 학습(NCS) + 안전(KOSHA) + 현장실습 등 복수 네임스페이스 참조 |
| 도메인 구조 전환 | 완전 대체 / 레이어 추가 | **레이어 추가** | 기존 도메인(namespace) 체계 위에 전공(major) 레이어를 추가하여 하위호환 유지 |
| 전공 설정 저장 | Python dict / JSON 파일 / DB | **Python dict** | 기존 domain_config.py 패턴 일관성, 배포 단순성 |
| 검색 전략 | 전공 네임스페이스 병합 검색 / 순차 검색 후 RRF | **병합 검색** | Pinecone filter로 복수 namespace 한 번에 검색 가능 |
| 프론트엔드 전공 UI | 별도 페이지 / 기존 domain.html 확장 | **domain.html 확장** | 기존 도메인별 동적 렌더링 구조를 전공 기반으로 전환 |
| LLM 프롬프트 관리 | 전공별 전체 프롬프트 / 템플릿 + 변수 | **템플릿 + 변수** | 공통 구조는 유지하면서 전공 특화 지식만 변수로 주입 |

### 6.3 아키텍처 전환 개념도

```
[현재 AS-IS: 토픽 기반]
─────────────────────────────────────────────
홈 → semiconductor-v2 (반도체 NCS)
   → laborlaw (노동법)
   → field-training (현장실습)
   → kosha (안전보건)
   → msds (화학물질)

각 도메인 독립 접근, 학생이 직접 선택

[목표 TO-BE: 전공 중심]
─────────────────────────────────────────────
홈 → 전공 선택 (반도체과 / 전기전자과 / 기계과 / ...)
        ↓
   전공 학습환경 (통합 검색 + AI Q&A)
        │
        ├── 전공 학습자료 (NCS) ← semiconductor-v2
        ├── 전공 안전자료 ← kosha (전공 필터링)
        ├── 현장실습 가이드 ← field-training (전공 필터링)
        └── 공통 자료 (자동 크로스서치)
             ├── 노동법 ← laborlaw
             ├── 화학물질 안전 ← msds
             └── 일반 안전보건 ← kosha (공통)

전공 추가 = MAJOR_CONFIG에 엔트리 추가 + 문서 업로드
```

### 6.4 핵심 데이터 구조 (안)

```python
# services/major_config.py (신규)
MAJOR_CONFIG = {
    "semiconductor": {
        "name": "반도체과",
        "icon": "microchip",
        "color": "#6366f1",
        "description": "반도체 제조공정, 장비, 품질관리",
        # 이 전공이 검색할 네임스페이스 목록
        "namespaces": {
            "primary": "semiconductor-v2",   # 전공 핵심 학습자료
            "safety": "kosha",               # 전공 관련 안전자료
            "training": "field-training",    # 현장실습 자료
        },
        # 안전자료 검색 시 전공 특화 필터 키워드
        "safety_keywords": ["반도체", "클린룸", "화학물질", "정전기", "CVD", "에칭"],
        # LLM 프롬프트에 주입될 전공 변수
        "prompt_context": {
            "specialty": "반도체 제조공정",
            "core_topics": ["CVD/PVD 공정", "리소그래피", "에칭", "패키징", "품질관리"],
            "safety_focus": ["화학물질 취급", "클린룸 안전", "정전기 방지"],
        },
        # 전공 진입 시 표시할 샘플 질문
        "sample_questions": [
            "CVD 공정의 원리와 종류를 설명해주세요",
            "클린룸에서의 안전수칙은 무엇인가요?",
            "반도체 현장실습 시 주의사항을 알려주세요",
        ],
        # 자동 라우팅 키워드 (query_router 확장)
        "routing_keywords": {
            "high": ["반도체", "웨이퍼", "CVD", "PVD", "에칭", "리소그래피"],
            "low": ["공정", "제조", "팹", "클린룸"],
        },
    },
    # 향후 추가 예시
    # "electrical": {
    #     "name": "전기전자과",
    #     "namespaces": {"primary": "electrical-v1", "safety": "kosha", ...},
    #     ...
    # },
}

# 전 전공 공통으로 크로스서치되는 네임스페이스
COMMON_NAMESPACES = {
    "laborlaw": {"name": "노동법", "description": "근로기준법, 4대보험, 최저임금"},
    "msds": {"name": "화학물질 안전", "description": "MSDS, GHS, 화학물질 취급"},
    "kosha-common": {"name": "일반 안전보건", "description": "공통 산업안전, 보호구, 응급처치"},
}
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` has coding conventions section
- [ ] `docs/01-plan/conventions.md` exists
- [ ] ESLint/Prettier (N/A — Python 프로젝트)
- [x] 기존 코딩 패턴: Flask Blueprint, 서비스 싱글톤, domain_config 패턴

### 7.2 Conventions to Define/Verify

| 카테고리 | 현재 상태 | 정의 필요 사항 | 우선순위 |
|----------|-----------|----------------|:--------:|
| **전공 설정 구조** | 없음 | MAJOR_CONFIG 딕셔너리 스키마 | High |
| **네임스페이스 네이밍** | 기존 규칙 있음 | 전공별 네임스페이스 네이밍 규칙 | High |
| **전공 추가 절차** | 없음 | 전공 추가 시 체크리스트 문서화 | Medium |
| **프롬프트 템플릿** | 도메인별 하드코딩 | 전공 변수 주입 템플릿 표준화 | Medium |

### 7.3 Environment Variables Needed

| 변수 | 용도 | 범위 | 신규 여부 |
|------|------|------|:---------:|
| `DEFAULT_MAJOR` | 기본 전공 (semiconductor) | Server | 신규 |
| 기존 변수 모두 유지 | — | — | 유지 |

---

## 8. Implementation Phases (권장 순서)

### Phase 1: 전공 설정 레이어 구축 (Core)
1. `services/major_config.py` 생성 — 전공 설정 데이터 구조
2. `services/domain_config.py` 리팩토링 — 전공 레이어와 기존 도메인 레이어 연결
3. 기존 5개 도메인을 반도체 전공 하위로 매핑

### Phase 2: RAG 파이프라인 전공 지원
4. `services/rag_pipeline.py` — 전공별 다중 네임스페이스 검색 지원
5. `services/query_router.py` — 전공 기반 라우팅 확장
6. 전공별 LLM 프롬프트 템플릿 시스템

### Phase 3: 프론트엔드 전공 UI
7. 홈 화면 전공 선택 카드 UI
8. `templates/domain.html` → 전공 기반 통합 학습환경으로 전환
9. 네비게이션 전공 전환 지원

### Phase 4: 문서 파이프라인 전공 태깅
10. CLI 문서 업로드 시 전공 메타데이터 부여
11. 전공 추가 플러그인 구조 검증 (더미 전공 테스트)

---

## 9. Next Steps

1. [ ] Design 문서 작성 (`vocational-major-restructuring.design.md`)
2. [ ] 전공 설정 데이터 구조 상세 설계
3. [ ] 기존 도메인 → 전공 매핑 테이블 확정
4. [ ] Implementation 시작

---

## Version History

| 버전 | 날짜 | 변경 사항 | 작성자 |
|------|------|-----------|--------|
| 0.1 | 2026-03-14 | 초안 작성 | zealnutkim |
