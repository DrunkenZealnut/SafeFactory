#!/usr/bin/env python3
"""
add_toc_summaries.py

meta.json의 table_of_contents 항목별로 해당 페이지 elements를 수집하고,
OpenAI로 2~3문장 요약문을 생성해 toc[i]["summary"]에 저장합니다.

markdown 파일 없이 meta.json만으로 동작합니다.

사용법:
    python scripts/add_toc_summaries.py --meta-file <경로>/_meta.json
    python scripts/add_toc_summaries.py --meta-file <경로>/_meta.json --overwrite
    python scripts/add_toc_summaries.py --meta-file <경로>/_meta.json --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


def collect_elements_for_section(
    elements: List[Dict],
    start_page: int,
    end_page: Optional[int],
    max_chars: int = 3000,
) -> str:
    """
    주어진 page 범위([start_page, end_page))에 속하는 elements의 텍스트를 수집합니다.

    Args:
        elements: meta.json의 elements 배열
        start_page: 섹션 시작 page_id (inclusive)
        end_page: 섹션 종료 page_id (exclusive). None이면 마지막까지.
        max_chars: 반환 텍스트 최대 길이 (LLM 입력 제한용)

    Returns:
        수집된 텍스트
    """
    texts = []
    total_len = 0

    for elem in elements:
        page_id = elem.get("metadata", {}).get("page_id")
        if page_id is None:
            continue
        if page_id < start_page:
            continue
        if end_page is not None and page_id >= end_page:
            continue

        elem_type = elem.get("type", "")
        # Image는 텍스트 내용이 없으므로 제외
        if elem_type == "Image":
            continue

        text = elem.get("text", "").strip()
        if not text:
            continue

        texts.append(text)
        total_len += len(text)
        if total_len >= max_chars:
            break

    combined = "\n".join(texts)
    return combined[:max_chars]


def generate_summary(client, section_title: str, content: str) -> Optional[str]:
    """
    OpenAI gpt-4o-mini를 사용해 섹션 요약문을 생성합니다.

    Args:
        client: openai.OpenAI 클라이언트
        section_title: 소주제 제목
        content: 섹션 내용 텍스트

    Returns:
        요약문 문자열 (실패 시 None)
    """
    if not content.strip():
        return None

    prompt = (
        f"다음은 '{section_title}' 소주제의 내용입니다.\n"
        "이 내용을 2~3문장으로 핵심만 요약해 주세요. "
        "전문 용어는 그대로 사용하고, 검색에 유용한 키워드를 포함하세요. "
        "한국어로 작성하세요.\n\n"
        f"내용:\n{content}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"  [경고] 요약 생성 실패: {e}", file=sys.stderr)
        return None


def add_toc_summaries(
    meta_file: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    delay: float = 0.5,
) -> None:
    """
    meta.json에 TOC 항목별 요약문을 추가합니다.

    Args:
        meta_file: meta.json 파일 경로
        overwrite: 이미 summary가 있는 항목도 덮어쓸지 여부
        dry_run: True면 파일을 실제로 저장하지 않고 출력만 함
        delay: API 호출 간 대기 시간 (초)
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[오류] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    print(f"파일 로드: {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    toc: List[Dict] = meta.get("table_of_contents", [])
    elements: List[Dict] = meta.get("elements", [])

    if not toc:
        print("[경고] table_of_contents가 비어 있습니다.")
        return

    print(f"TOC 항목 수: {len(toc)}, elements 수: {len(elements)}")

    skipped = 0
    generated = 0
    failed = 0

    for i, entry in enumerate(toc):
        title = entry.get("title", "").strip()
        start_page = entry.get("page_id")

        if start_page is None:
            print(f"  [{i+1}/{len(toc)}] '{title}' — page_id 없음, 건너뜀")
            skipped += 1
            continue

        # 이미 요약이 있고 overwrite=False면 건너뜀
        existing_summary = entry.get("summary", "").strip()
        if existing_summary and not overwrite:
            print(f"  [{i+1}/{len(toc)}] '{title}' — 이미 요약 있음, 건너뜀")
            skipped += 1
            continue

        # 다음 TOC 항목의 page_id를 end_page로 사용
        end_page = None
        if i + 1 < len(toc):
            end_page = toc[i + 1].get("page_id")

        # 해당 섹션 elements 수집
        content = collect_elements_for_section(elements, start_page, end_page)

        if not content:
            print(f"  [{i+1}/{len(toc)}] '{title}' — 내용 없음, 건너뜀")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(toc)}] '{title}' (page {start_page}~{end_page or '끝'}) 요약 생성 중...")

        if dry_run:
            print(f"    [dry-run] 내용 미리보기: {content[:100]}...")
            generated += 1
            continue

        summary = generate_summary(client, title, content)
        if summary:
            entry["summary"] = summary
            print(f"    → {summary[:80]}{'...' if len(summary) > 80 else ''}")
            generated += 1
        else:
            failed += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\n완료: 생성 {generated}개, 건너뜀 {skipped}개, 실패 {failed}개")

    if dry_run:
        print("[dry-run] 파일을 저장하지 않습니다.")
        return

    if generated > 0:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {meta_file}")
    else:
        print("변경 사항 없음, 파일을 저장하지 않습니다.")


def find_meta_files(data_dir: Path) -> List[Path]:
    """
    data_dir 하위에서 처리 대상 *_meta.json 파일을 찾습니다.
    _marker_meta.json 및 new_meta.json, report 폴더는 제외합니다.
    """
    candidates = []
    for path in sorted(data_dir.rglob("*_meta.json")):
        name = path.name
        if name.endswith("_marker_meta.json"):
            continue
        if name == "new_meta.json":
            continue
        if "report" in path.parts:
            continue
        candidates.append(path)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="meta.json TOC 항목별 요약문 자동 생성"
    )
    # 단일 파일 또는 디렉터리 중 하나를 지정
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--meta-file",
        type=Path,
        help="처리할 _meta.json 파일 경로 (단일)",
    )
    group.add_argument(
        "--data-dir",
        type=Path,
        help="하위 *_meta.json 파일을 모두 처리할 루트 디렉터리",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 summary가 있는 항목도 덮어씀 (기본: 건너뜀)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 저장 없이 동작 확인만 함",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="API 호출 간 대기 시간(초), 기본: 0.3",
    )
    args = parser.parse_args()

    if args.meta_file:
        if not args.meta_file.exists():
            print(f"[오류] 파일을 찾을 수 없습니다: {args.meta_file}", file=sys.stderr)
            sys.exit(1)
        meta_files = [args.meta_file]
    else:
        if not args.data_dir.is_dir():
            print(f"[오류] 디렉터리를 찾을 수 없습니다: {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        meta_files = find_meta_files(args.data_dir)
        print(f"처리 대상 파일: {len(meta_files)}개\n")

    for idx, meta_file in enumerate(meta_files, 1):
        if len(meta_files) > 1:
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(meta_files)}] {meta_file.name}")
            print(f"{'='*60}")
        add_toc_summaries(
            meta_file=meta_file,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
