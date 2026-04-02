"""LLM-as-Judge answer quality evaluation.

Scores generated answers on 4 dimensions using an LLM evaluator:
- Accuracy: Are the facts in the answer correct and supported by sources?
- Relevance: Does the answer actually address the question?
- Readability: Is the answer understandable by a 직업계고 student (age 16-18)?
- Citation Quality: Are sources properly cited and do citations support claims?

Usage:
    from scripts.eval.llm_judge import judge_answer
    result = judge_answer(query, answer, sources_text, expected_facts)
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Scoring rubric (in Korean for consistency with the domain)
JUDGE_PROMPT = """당신은 RAG 시스템의 답변 품질을 평가하는 전문 평가자입니다.
직업계고 학생(16~18세)을 위한 반도체/산업안전 교육 플랫폼의 AI 답변을 채점합니다.

## 평가 기준

### 1. 정확성 (Accuracy) [1-5]
- 5: 모든 정보가 제공된 출처와 일치하며 사실적으로 정확함
- 4: 대부분 정확하지만 사소한 부정확 1개
- 3: 핵심 내용은 맞지만 일부 부정확하거나 오해의 소지
- 2: 중요한 사실 오류 포함
- 1: 대부분 부정확하거나 출처와 모순됨

### 2. 관련성 (Relevance) [1-5]
- 5: 질문에 완벽하게 답변하며 핵심 정보 모두 포함
- 4: 질문에 잘 답변하지만 일부 핵심 정보 누락
- 3: 부분적으로 관련있지만 질문의 핵심을 놓침
- 2: 질문과 약간만 관련되는 내용
- 1: 질문과 무관한 답변

### 3. 가독성 (Readability) [1-5]
- 5: 고1 학생도 쉽게 이해할 수 있는 명확한 설명, 전문용어에 설명 병기
- 4: 대체로 이해하기 쉽지만 일부 설명이 부족한 전문용어 존재
- 3: 보통 수준, 일부 어려운 부분이 있음
- 2: 전문용어 과다하고 설명 부족
- 1: 이해하기 매우 어려움

### 4. 인용 품질 (Citation) [1-5]
- 5: 모든 주장에 적절한 출처 인용 [N], 인용이 실제 내용과 일치
- 4: 대부분 인용 있으나 일부 주장에 인용 누락
- 3: 인용이 있지만 불규칙적
- 2: 인용이 거의 없거나 부정확한 인용 번호
- 1: 인용 전혀 없음

### 5. 환각 감지 (Hallucination) [yes/no]
- yes: 출처에 없는 내용을 사실인 것처럼 서술함
- no: 모든 내용이 출처에 기반하거나 명확히 일반 지식으로 표시됨

## 입력

**질문**: {query}

**AI 답변**:
{answer}

**제공된 출처 요약**:
{sources}

{expected_section}

## 출력 형식 (반드시 JSON으로)

```json
{{
  "accuracy": <1-5>,
  "relevance": <1-5>,
  "readability": <1-5>,
  "citation": <1-5>,
  "hallucination": <true/false>,
  "overall": <1-5>,
  "brief_reason": "<1줄 한국어 요약>"
}}
```

JSON만 출력하세요. 다른 텍스트는 포함하지 마세요."""


def judge_answer(
    query: str,
    answer: str,
    sources_text: str,
    expected_facts: Optional[List[str]] = None,
    model: str = 'gemini-2.5-flash',
) -> Dict:
    """Score an answer using LLM-as-judge.

    Args:
        query: The original question.
        answer: The generated answer to evaluate.
        sources_text: Summary of source documents provided to the LLM.
        expected_facts: Optional list of facts the answer should contain.
        model: LLM model to use for judging.

    Returns:
        Dict with scores: accuracy, relevance, readability, citation, hallucination,
        overall, brief_reason. Returns error dict on failure.
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')

    expected_section = ''
    if expected_facts:
        facts_str = ', '.join(expected_facts)
        expected_section = f"**답변에 포함되어야 할 핵심 사실**: {facts_str}"

    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer[:3000],  # Truncate very long answers
        sources=sources_text[:2000],
        expected_section=expected_section,
    )

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=500,
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        raw = resp.text or ''

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("Judge returned no JSON: %s", raw[:200])
            return {'error': 'no_json', 'raw': raw[:200]}

        scores = json.loads(json_match.group())

        # Validate and clamp scores
        for key in ('accuracy', 'relevance', 'readability', 'citation', 'overall'):
            if key in scores:
                scores[key] = max(1, min(5, int(scores[key])))

        if 'hallucination' in scores:
            scores['hallucination'] = bool(scores['hallucination'])

        return scores

    except Exception as e:
        logger.error("Judge failed: %s", e)
        return {'error': str(e)}


def judge_batch(results: List[Dict], max_queries: int = 0) -> List[Dict]:
    """Run LLM-as-judge on a list of eval results that have answer text.

    Args:
        results: List of eval result dicts (must have 'answer_text' key).
        max_queries: Max queries to judge (0 = all). Useful for cost control.

    Returns:
        List of dicts with query_id and judge scores.
    """
    judged = []
    items = [r for r in results if r.get('answer_text')]
    if max_queries > 0:
        items = items[:max_queries]

    for i, r in enumerate(items):
        logger.info("[Judge %d/%d] %s", i + 1, len(items), r.get('id', '?'))
        scores = judge_answer(
            query=r.get('query', ''),
            answer=r.get('answer_text', ''),
            sources_text=r.get('sources_text', ''),
            expected_facts=r.get('expected_answer_contains', []),
        )
        scores['id'] = r.get('id', '')
        judged.append(scores)

    return judged


def print_judge_report(judged: List[Dict]):
    """Print summary of LLM-as-judge results."""
    valid = [j for j in judged if 'error' not in j]
    if not valid:
        print("No valid judge results.")
        return

    print("\n" + "=" * 60)
    print("LLM-as-Judge Answer Quality Report")
    print("=" * 60)
    print(f"Evaluated: {len(valid)} answers\n")

    for key in ('accuracy', 'relevance', 'readability', 'citation', 'overall'):
        vals = [j[key] for j in valid if key in j]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {key:12s}: {avg:.2f}/5  (min={min(vals)}, max={max(vals)})")

    hallucinations = sum(1 for j in valid if j.get('hallucination'))
    print(f"\n  Hallucinations: {hallucinations}/{len(valid)} ({hallucinations/len(valid):.0%})")

    # Worst scoring answers
    worst = sorted(valid, key=lambda j: j.get('overall', 5))[:3]
    if worst:
        print("\n  Lowest scoring answers:")
        for j in worst:
            print(f"    [{j.get('id', '?')}] overall={j.get('overall', '?')}/5 — {j.get('brief_reason', '')}")

    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LLM-as-Judge standalone evaluation')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to eval results JSON (from eval_pipeline.py --eval-answer)')
    parser.add_argument('--max', type=int, default=0, help='Max queries to judge (0=all)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    judged = judge_batch(results, max_queries=args.max)
    print_judge_report(judged)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(judged, f, ensure_ascii=False, indent=2)
        print(f"\nJudge results saved to: {args.output}")
