# Pinecone Agent

폴더 내 파일(이미지, 마크다운, JSON)을 처리하여 Pinecone 벡터 데이터베이스에 저장하는 에이전트입니다.

## 기능

- **이미지 처리**: OpenAI Vision API를 사용하여 이미지 내용을 텍스트로 설명
- **마크다운 처리**: 시맨틱 청킹으로 의미 단위 분할
- **JSON 처리**: 배열/객체 구조에 맞게 지능적 분할
- **벡터 임베딩**: OpenAI text-embedding-3-small 모델 사용
- **Pinecone 저장**: 자동 인덱스 생성 및 배치 업로드

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 입력
```

## 환경변수 설정

`.env` 파일에 다음 정보를 입력하세요:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

## 사용법

### 1. 폴더 처리 및 업로드

```bash
# 기본 사용
python main.py process ./my_documents

# 네임스페이스 지정
python main.py process ./my_documents --namespace project-docs

# 하위 폴더 미포함
python main.py process ./my_documents --no-recursive

# 청크 크기 조절
python main.py process ./my_documents --max-chunk-tokens 300
```

### 2. 검색

```bash
# 기본 검색
python main.py search "반도체 제조 공정"

# 결과 개수 지정
python main.py search "리소그래피 재료" --top-k 10

# 파일 타입 필터
python main.py search "공정 다이어그램" --filter-file-type image
```

### 3. 통계 확인

```bash
python main.py stats
```

### 4. 벡터 삭제

```bash
# 특정 파일의 벡터 삭제
python main.py delete --source-file "/path/to/file.md"

# 전체 삭제 (주의!)
python main.py delete --all
```

## Python에서 직접 사용

```python
from src.agent import PineconeAgent

# 에이전트 초기화
agent = PineconeAgent(
    openai_api_key="your-openai-key",
    pinecone_api_key="your-pinecone-key",
    pinecone_index_name="my-index"
)

# 폴더 처리
result = agent.process_folder(
    folder_path="./documents",
    namespace="my-docs",
    verbose=True
)

print(f"처리된 파일: {result.processed_files}")
print(f"업로드된 벡터: {result.uploaded_vectors}")

# 검색
results = agent.search(
    query="반도체 공정",
    top_k=5
)

for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Content: {r['metadata']['content'][:200]}")
```

## 지원 파일 형식

| 형식 | 확장자 | 처리 방식 |
|------|--------|-----------|
| 이미지 | .png, .jpg, .jpeg, .gif, .bmp, .webp | Vision API로 설명 생성 |
| 마크다운 | .md, .markdown | 시맨틱 청킹 |
| JSON | .json | 구조 기반 분할 |

## 프로젝트 구조

```
pinecone_agent/
├── main.py              # CLI 인터페이스
├── requirements.txt     # 의존성
├── .env.example         # 환경변수 예시
├── README.md
└── src/
    ├── __init__.py
    ├── agent.py             # 메인 에이전트
    ├── file_loader.py       # 파일 로딩
    ├── image_describer.py   # 이미지 설명 생성
    ├── semantic_chunker.py  # 시맨틱 청킹
    ├── embedding_generator.py  # 임베딩 생성
    └── pinecone_uploader.py    # Pinecone 업로드
```

## 라이선스

MIT License
