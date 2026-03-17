# Footer 제작 Design

> Plan 참조: `docs/01-plan/features/footer-redesign.plan.md`

## 1. 변경 범위

| 파일 | 작업 | 변경 유형 |
|------|------|-----------|
| `templates/base.html` | footer CSS + HTML 추가 | 추가 |
| `templates/home.html` | 기존 footer HTML 제거, footer CSS 제거 | 삭제 |

---

## 2. HTML 구조

### 2.1 base.html에 추가할 Footer HTML

`</main>` 태그 바로 뒤, `<script>` 앞에 삽입:

```html
<!-- Footer -->
<footer class="sf-footer">
    <div class="sf-footer-inner">
        <div class="sf-footer-org">
            <span class="sf-footer-org-name">청년노동자인권센터</span>
            <span class="sf-footer-org-detail">서울시 종로구 성균관로12 5층</span>
            <span class="sf-footer-org-detail">
                <a href="mailto:admin@younglabor.kr">admin@younglabor.kr</a>
            </span>
        </div>
        <div class="sf-footer-links">
            <a href="#">이용약관</a>
            <a href="#">개인정보처리방침</a>
            <a href="#">고객센터</a>
        </div>
        <p class="sf-footer-copy">&copy; {{ now().year }} SafeFactory. AI 기반 산업안전 지식 플랫폼</p>
    </div>
</footer>
```

### 2.2 DOM 구조 트리

```
footer.sf-footer
  └─ div.sf-footer-inner
       ├─ div.sf-footer-org            ← 신규: 단체 정보 영역
       │    ├─ span.sf-footer-org-name   ← 단체명 (볼드)
       │    ├─ span.sf-footer-org-detail ← 주소
       │    └─ span.sf-footer-org-detail ← 이메일 (mailto 링크)
       ├─ div.sf-footer-links           ← 기존 유지: 이용약관/개인정보/고객센터
       └─ p.sf-footer-copy              ← 기존 유지: 저작권
```

---

## 3. CSS 설계

### 3.1 base.html `<style>` 블록에 추가할 스타일

기존 프로젝트 CSS 변수(`theme.css`)를 활용하여 일관성 유지.

```css
/* ===== Footer ===== */
.sf-footer {
    max-width: 960px;
    margin: 0 auto;
    padding: 0 24px;
}
.sf-footer-inner {
    border-top: 1px solid var(--sf-border);
    padding: 28px 0;
    text-align: center;
}
.sf-footer-org {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    margin-bottom: 16px;
}
.sf-footer-org-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--sf-text-1);
}
.sf-footer-org-detail {
    font-size: 12px;
    color: var(--sf-text-3);
}
.sf-footer-org-detail a {
    color: var(--sf-text-3);
    text-decoration: none;
    transition: color 0.2s;
}
.sf-footer-org-detail a:hover {
    color: var(--sf-purple);
}
.sf-footer-links {
    margin-bottom: 10px;
}
.sf-footer-links a {
    font-size: 12px;
    color: var(--sf-text-4);
    text-decoration: none;
    margin: 0 12px;
    transition: color 0.2s;
}
.sf-footer-links a:hover {
    color: var(--sf-purple-light);
}
.sf-footer-copy {
    font-size: 12px;
    color: var(--sf-text-3);
}
```

### 3.2 반응형 대응

```css
@media (max-width: 600px) {
    .sf-footer { padding: 0 16px; }
}
```

### 3.3 디자인 토큰 매핑

| 요소 | CSS 변수 | 값 |
|------|----------|-----|
| 단체명 색상 | `--sf-text-1` | #0f172a |
| 주소/이메일 색상 | `--sf-text-3` | #94a3b8 |
| 링크 색상 | `--sf-text-4` | #cbd5e1 |
| 링크 hover | `--sf-purple-light` | #475569 |
| 이메일 hover | `--sf-purple` | #1e293b |
| 구분선 | `--sf-border` | #e2e8f0 |
| 최대 너비 | 960px | 프로젝트 컨벤션 |

---

## 4. home.html 제거 대상

### 4.1 HTML 제거 (line 677-686)
```html
<!-- Footer -->  ← 전체 삭제
<footer class="sf-footer">...</footer>
```

### 4.2 CSS 제거 (head_extra 블록 내, line 490-515)
```css
/* ===== Footer ===== */   ← 전체 삭제
.sf-footer { ... }
.sf-footer-inner { ... }
.sf-footer-links { ... }
.sf-footer-links a { ... }
.sf-footer-links a:hover { ... }
.sf-footer-copy { ... }
```

---

## 5. 구현 순서 (체크리스트)

1. [ ] `base.html` — `{% block head_extra %}` 앞에 footer CSS 추가
2. [ ] `base.html` — `</main>` 뒤에 footer HTML 추가
3. [ ] `home.html` — footer CSS 블록 제거 (line 490-515)
4. [ ] `home.html` — footer HTML 제거 (line 677-686)
5. [ ] 동작 확인: 모든 페이지에 footer 표시되는지 확인

---

## 6. 영향 분석

| 페이지 | 변경 전 | 변경 후 |
|--------|---------|---------|
| `/` (홈) | footer 있음 (home.html 자체) | footer 있음 (base.html 공통) |
| `/learn` | footer 없음 | footer 표시 |
| `/msds` | footer 없음 | footer 표시 |
| `/questions` | footer 없음 | footer 표시 |
| `/community` | footer 없음 | footer 표시 |
| `/news` | footer 없음 | footer 표시 |
| `/login` | footer 없음 | footer 표시 |
| `/admin` | footer 없음 | footer 표시 |
| `/mypage` | footer 없음 | footer 표시 |

> `sf-page-content`의 `min-height: calc(100vh - 65px)`는 footer가 `main` 바깥이므로 자연스럽게 footer가 콘텐츠 아래에 위치함. 수정 불필요.
