# SafeFactory.kr 네이티브 앱 + 웹 리뉴얼 통합 로드맵

> 작성일: 2026-02-14
> 상태: 초안 (전체 메뉴/스타일 반영하여 재기획 예정)

## Context

safefactory.kr은 현재 Flask 기반 서버 렌더링 웹앱으로, 4개 도메인(반도체/노동법/현장실습/MSDS)에 걸친 AI 문서 검색 및 RAG 질의응답 시스템이다. **React Native (Expo) + React Native Web**으로 iOS/Android/Web 3개 플랫폼을 하나의 코드베이스로 통합 개발한다. 카메라 촬영 MSDS 화학물질 식별을 핵심 네이티브 기능으로 우선 개발한다.

### 확정된 기술 결정
- **프레임워크**: React Native (Expo)
- **타겟 플랫폼**: iOS + Android 동시 + 웹 리뉴얼 (React Native Web)
- **웹사이트 전략**: 기존 Flask 템플릿 → React Native Web으로 통합 리뉴얼
- **우선 네이티브 기능**: 카메라 촬영 MSDS 화학물질 식별

---

## 현재 시스템 분석

### 아키텍처 현황
```
[Flask 서버] ─── 템플릿 렌더링 ──→ [브라우저]
     │
     ├── /api/search (POST)      벡터/하이브리드/키워드 검색
     ├── /api/ask (POST)         RAG 질의응답
     ├── /api/ask/stream (POST)  SSE 스트리밍 응답
     ├── /api/stats              인덱스 통계
     ├── /api/namespaces         네임스페이스 목록
     ├── /api/msds/*             MSDS 화학물질 검색
     └── /api/sources            소스 자동완성
```

### 핵심 기능 목록
| 기능 | 복잡도 | 네이티브 전환 고려사항 |
|------|--------|----------------------|
| 4개 도메인 검색 | 중 | 도메인별 UI 테마 분리 필요 |
| SSE 스트리밍 응답 | 상 | EventSource 또는 WebSocket 전환 |
| Markdown 렌더링 | 중 | 네이티브 Markdown 라이브러리 필요 |
| 차트 시각화 (Chart.js) | 중 | 네이티브 차트 라이브러리 전환 |
| 이미지 갤러리/라이트박스 | 하 | 네이티브 이미지 뷰어 활용 |
| 임금/보험 계산기 | 중 | 폼 UI + API 연동 |
| MSDS 화학물질 검색 | 중 | 카메라 촬영 → 이미지 식별 강화 가능 |
| 파일 필터링 (@멘션) | 하 | 텍스트 입력 파싱 동일 |

---

## 기술 스택

### 전체 기술 스택
```
┌─────────────────────────────────────────────────────┐
│  통합 프론트엔드 (React Native + Expo + RN Web)     │
│  ├── TypeScript                                      │
│  ├── Expo Router (파일 기반 라우팅, 웹/앱 공유)      │
│  ├── NativeWind v4 (Tailwind CSS for RN + Web)       │
│  ├── React Query (TanStack Query - API 캐싱)         │
│  ├── Zustand (경량 상태관리)                         │
│  ├── @expo/html-elements (웹 SEO용 시맨틱 태그)      │
│  ├── react-native-markdown-display (앱)              │
│  ├── react-markdown (웹 폴백)                        │
│  ├── victory-native (차트)                           │
│  ├── expo-camera (MSDS 화학물질 촬영)                │
│  ├── expo-image (최적화된 이미지 로딩)               │
│  └── expo-image-picker (갤러리 선택)                 │
├─────────────────────────────────────────────────────┤
│  백엔드 API (기존 Flask 유지 + 개선)                │
│  ├── API 버전관리 (/api/v1/)                         │
│  ├── JWT 인증 (앱) + 세션 인증 (웹) 이중 지원        │
│  ├── CORS 설정 (앱 번들ID + 웹 도메인)               │
│  ├── Rate Limiting                                   │
│  └── SSE 스트리밍 유지 (앱/웹 모두 호환)             │
├─────────────────────────────────────────────────────┤
│  배포 인프라                                         │
│  ├── EAS Build (iOS/Android 빌드)                    │
│  ├── Expo Web → Vercel/Netlify (웹 호스팅)           │
│  ├── 기존 서버 (API 백엔드 유지)                     │
│  └── safefactory.kr → 웹 빌드 배포                   │
└─────────────────────────────────────────────────────┘
```

### 플랫폼별 코드 공유 전략
```
코드 공유율 목표: ~90%
├── 공유 코드 (iOS + Android + Web)
│   ├── 비즈니스 로직 (services/, stores/, hooks/)
│   ├── 도메인 설정 (constants/)
│   ├── 타입 정의 (types/)
│   └── 대부분의 UI 컴포넌트 (components/)
│
├── 플랫폼 분기 (Platform.select / .native.tsx / .web.tsx)
│   ├── 카메라 (앱: expo-camera, 웹: <input type="file">)
│   ├── Markdown 렌더링 (앱: RN Markdown, 웹: react-markdown)
│   ├── 스트리밍 (앱: react-native-sse, 웹: EventSource)
│   └── 이미지 뷰어 (앱: 네이티브 줌, 웹: CSS 모달)
│
└── 플랫폼 전용
    ├── iOS/Android: 푸시 알림, 생체인증, 음성검색
    └── Web: SEO 메타태그, 서버사이드 렌더링(선택)
```

---

## 단계별 로드맵

### Phase 0: API 계층 분리 및 정비 (2~3주)
> 네이티브 앱 개발 전 백엔드 API를 앱 친화적으로 정비

**목표**: 프론트엔드-독립적인 API 서버 확립

**작업 항목**:
1. **API 버전관리 도입**
   - 모든 엔드포인트에 `/api/v1/` 접두사 적용
   - 기존 웹 프론트엔드는 v1 API로 전환

2. **인증 시스템 추가**
   - JWT 기반 인증 (앱용)
   - API Key 인증 (서버간 통신용)
   - 게스트 모드 (제한된 쿼리 수)

3. **API 응답 표준화**
   ```json
   {
     "status": "success|error",
     "data": { ... },
     "meta": { "timestamp": "...", "version": "v1" }
   }
   ```

4. **CORS 설정 및 Rate Limiting**
   - 앱 도메인/번들ID 허용
   - 사용자별 요청 제한 (분당 30회)

5. **스트리밍 프로토콜 정비**
   - SSE 유지 (React Native에서도 지원)
   - 선택적 WebSocket 엔드포인트 추가

**수정 파일**: `web_app.py` (API 라우트 분리)

---

### Phase 1: Expo 프로젝트 초기 구축 (3~4주)
> 앱+웹 통합 프로젝트 기본 구조 및 핵심 UI 프레임 구축

**작업 항목**:

1. **프로젝트 초기화** (`npx create-expo-app safefactory-app --template tabs`)
   ```
   safefactory-app/
   ├── app/                        # Expo Router 페이지 (웹+앱 공유)
   │   ├── _layout.tsx             # 루트 레이아웃
   │   ├── (tabs)/                 # 탭 네비게이션
   │   │   ├── _layout.tsx         # 탭 레이아웃 (앱: 하단탭, 웹: 상단 네비)
   │   │   ├── index.tsx           # 홈 (도메인 선택)
   │   │   ├── search.tsx          # 검색 탭
   │   │   └── settings.tsx        # 설정
   │   ├── domain/
   │   │   └── [id].tsx            # 도메인별 검색 화면
   │   ├── msds/
   │   │   ├── index.tsx           # MSDS 검색
   │   │   ├── camera.tsx          # MSDS 카메라 촬영 (★ 우선)
   │   │   └── [chemicalId].tsx    # MSDS 상세
   │   └── calculator/
   │       └── index.tsx           # 임금/보험 계산기
   ├── components/
   │   ├── ui/                     # 공통 UI (Button, Card, Input...)
   │   ├── domain/                 # 도메인별 컴포넌트
   │   ├── search/                 # 검색 관련
   │   ├── streaming/              # 스트리밍 응답 렌더러
   │   │   ├── StreamingView.tsx           # 공유 로직
   │   │   ├── MarkdownRenderer.native.tsx # 앱용
   │   │   └── MarkdownRenderer.web.tsx    # 웹용
   │   └── camera/                 # 카메라 관련
   │       ├── CameraCapture.native.tsx    # 앱: expo-camera
   │       └── CameraCapture.web.tsx       # 웹: file input
   ├── services/
   │   ├── api.ts                  # API 클라이언트 (React Query)
   │   ├── streaming.ts            # SSE 스트리밍 파서
   │   │   ├── streaming.native.ts # react-native-sse
   │   │   └── streaming.web.ts    # EventSource
   │   └── auth.ts                 # JWT 인증
   ├── stores/                     # Zustand 상태관리
   ├── hooks/                      # 커스텀 훅
   ├── constants/domains.ts        # 도메인 설정 (색상, 아이콘, 질문)
   └── types/                      # TypeScript 타입
   ```

2. **웹 빌드 설정** (`app.json` + `metro.config.js`)
   - `expo export:web` → 정적 빌드 (safefactory.kr 배포)
   - 또는 Next.js + Expo 통합 (SSR 필요 시)
   - NativeWind v4 웹 호환 설정

3. **도메인 설정 시스템** (`constants/domains.ts`)
   - 기존 `DOMAIN_CONFIG` 딕셔너리를 TypeScript로 포팅
   - 도메인별 색상, 아이콘, 샘플 질문, 기능 태그

4. **API 클라이언트** (`services/api.ts`)
   - ky 기반 HTTP 클라이언트 (경량, fetch 기반, 웹+앱 호환)
   - React Query 통합 (캐싱, 재시도, 무효화)
   - 플랫폼별 SSE 스트리밍 파서
   - 오프라인 큐잉 (앱 전용)

5. **홈 화면** (도메인 선택 그리드)
   - 4개 도메인 카드 (아이콘, 설명, 기능 태그)
   - 실시간 통계 패널
   - 반응형: 앱(2열 그리드) / 웹(4열 그리드)

---

### Phase 2: 핵심 검색/RAG 화면 구현 (4~5주)
> 앱의 핵심 기능인 AI 검색 및 응답 화면

**작업 항목**:

1. **검색 입력 컴포넌트**
   - 텍스트 입력 + 음성 입력 (Expo Speech)
   - @멘션 자동완성
   - 검색 모드 선택 (벡터/하이브리드/키워드)
   - Top-K 슬라이더
   - 도메인별 필터 드롭다운

2. **스트리밍 응답 뷰**
   - SSE 이벤트 수신 및 파싱
   - 실시간 Markdown 렌더링 (타이핑 효과)
   - 인용 링크 [1][2] 처리
   - 테이블, 코드블록, 리스트 스타일링

3. **소스 레퍼런스 카드**
   - 점수 배지 (그라데이션)
   - 메타데이터 태그 (NCS 카테고리, 법조항 등)
   - 콘텐츠 미리보기
   - 탭하여 전체 내용 확장

4. **이미지 갤러리**
   - 관련 이미지 그리드
   - 풀스크린 이미지 뷰어 (핀치 줌, 스와이프)
   - 이미지 저장/공유

5. **차트 시각화**
   - victory-native로 차트 렌더링
   - 바, 파이, 라인 차트 지원
   - 인터랙티브 툴팁

---

### Phase 3: 도메인별 특화 기능 (3~4주)
> 카메라 MSDS 식별 우선 → 각 도메인 고유 기능 구현

**3-1. MSDS 카메라 식별 (★ 최우선)**
- **카메라 촬영 → 화학물질 즉시 식별** (`app/msds/camera.tsx`)
  - 앱: `expo-camera`로 실시간 카메라 뷰 + 촬영
  - 앱: `expo-image-picker`로 갤러리 이미지 선택
  - 웹: `<input type="file" accept="image/*" capture="environment">`
  - 촬영 이미지 → GPT-4o-mini Vision API → 화학물질명/CAS# 추출
  - 추출 결과 → KOSHA MSDS API 자동 검색 → 16개 섹션 상세 표시
- 화학물질 검색 (이름, CAS#, UN#)
- 위험등급별 색상 코딩 (GHS 표지 색상)
- **긴급 연락처 원탭 전화** (`Linking.openURL('tel:')`)
- 최근 조회 이력 로컬 저장

**3-2. 노동법 도메인**
- 임금 계산기 네이티브 폼 (슬라이더, 피커)
- 4대보험 계산기
- 계산 결과 시각화 (원형 차트)
- 계산 결과 공유/저장/PDF 내보내기

**3-3. 반도체/현장실습 도메인**
- 도메인별 커스텀 필터 UI
- 즐겨찾기 및 검색 기록
- 관련 이미지 핀치 줌 갤러리

---

### Phase 4: 네이티브 강화 기능 (3~4주)
> 웹에서는 불가능한 네이티브 앱만의 차별화 기능

**작업 항목**:

1. **오프라인 지원**
   - 최근 검색 결과 로컬 캐싱 (SQLite/MMKV)
   - 자주 사용하는 MSDS 데이터 다운로드
   - 계산기 오프라인 동작

2. **푸시 알림**
   - 법률 개정 알림 (노동법)
   - MSDS 업데이트 알림
   - 새 문서 추가 알림

3. **생체인증 & 보안**
   - Face ID / 지문인증 (expo-local-authentication)
   - 민감 데이터 암호화 저장 (expo-secure-store)

4. **음성 검색**
   - expo-speech로 음성 → 텍스트
   - 현장에서 손이 자유롭지 않을 때 유용

5. **위젯 (iOS/Android)**
   - 빠른 검색 위젯
   - 최근 검색 결과 위젯

6. **딥링크 & 공유**
   - 검색 결과 공유 링크
   - 앱 내 특정 화면으로 직접 이동
   - Universal Links (iOS) / App Links (Android)

---

### Phase 5: 테스트, 최적화, 배포 (2~3주)

**작업 항목**:

1. **테스트**
   - Jest + React Native Testing Library (단위/통합)
   - Detox E2E 테스트 (주요 사용자 플로우)
   - 성능 프로파일링 (Flipper)

2. **최적화**
   - 앱 번들 크기 최소화
   - 이미지 최적화 (WebP, 캐싱)
   - 리스트 가상화 (FlatList 최적화)
   - 메모리 누수 점검

3. **3개 플랫폼 배포**
   - **iOS**: Apple App Store 등록 (개발자 계정 $99/년, 심사 대응)
   - **Android**: Google Play Store 등록 (개발자 계정 $25 일회)
   - **웹**: `npx expo export:web` → safefactory.kr 배포 (Vercel 또는 기존 서버)
   - EAS Build (Expo Application Services) 활용
   - OTA 업데이트 설정 (expo-updates) - 앱 심사 없이 JS 번들 업데이트
   - safefactory.kr DNS를 새 웹 빌드로 전환 (기존 Flask는 API 전용)

4. **모니터링**
   - Sentry 에러 트래킹
   - Analytics (Firebase / Amplitude)
   - 성능 모니터링 (앱 시작 시간, API 응답 시간)

---

## 전체 타임라인 요약

```
Phase 0  [API 정비]          ████░░░░░░░░░░░░░░░░░░  2~3주
Phase 1  [프로젝트 구축]     ░░░░████░░░░░░░░░░░░░░  3~4주
Phase 2  [핵심 검색/RAG]     ░░░░░░░░██████░░░░░░░░  4~5주
Phase 3  [도메인 특화]       ░░░░░░░░░░░░░░████░░░░  3~4주
Phase 4  [네이티브 강화]     ░░░░░░░░░░░░░░░░░████░  3~4주
Phase 5  [테스트/배포]       ░░░░░░░░░░░░░░░░░░░░██  2~3주
                              ─────────────────────────
                              총 예상: 17~23주 (약 4~6개월)
```

---

## 위험 요소 및 대응

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|----------|
| SSE 스트리밍 React Native 호환 | 중 | 상 | `react-native-sse` 또는 `fetch` + ReadableStream 사용, WebSocket 폴백 |
| 앱스토어 심사 거절 | 중 | 상 | Apple 가이드라인 사전 검토, 최소 네이티브 기능 확보 |
| 대용량 Markdown 렌더링 성능 | 하 | 중 | 청크 단위 점진적 렌더링, 가상화 적용 |
| API 서버 부하 증가 | 중 | 중 | CDN 캐싱, Rate Limiting, 오프라인 캐시 활용 |

---

## 검증 방법

1. **Phase 0 완료 후**: Postman/Insomnia로 모든 API v1 엔드포인트 테스트
2. **Phase 1 완료 후**: Expo Go 앱으로 홈 화면 + 네비게이션 실기기 테스트
3. **Phase 2 완료 후**: 실제 쿼리로 스트리밍 응답 수신 및 렌더링 검증
4. **Phase 3 완료 후**: 각 도메인별 기능 통합 테스트 (계산기, MSDS 카메라 등)
5. **Phase 4 완료 후**: 오프라인 모드, 푸시 알림, 음성 검색 E2E 테스트
6. **Phase 5 완료 후**: TestFlight(iOS) / 내부 테스트(Android) 배포 후 실사용자 피드백

---

## TODO (재기획 시 반영할 사항)

- [ ] 전체 메뉴 구조 상세 설계 (IA: Information Architecture)
- [ ] 디자인 시스템 및 스타일 가이드 정의
- [ ] 각 화면별 와이어프레임/목업
- [ ] 사용자 플로우 다이어그램
- [ ] 도메인 추가/삭제 시 확장성 설계
