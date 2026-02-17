"""
LM Studio API 테스트 스크립트
"""
import os
import requests
import json
import time

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")

def test_lm_studio(prompt: str, description: str = "", max_tokens: int = 300):
    """LM Studio API 테스트"""
    print(f"\n{'='*60}")
    print(f"테스트: {description}")
    print(f"{'='*60}")
    print(f"질문: {prompt}")
    print(f"\n응답 대기 중...")

    start_time = time.time()

    try:
        response = requests.post(
            LM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens
            },
            timeout=60
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content']

            # 통계 정보
            usage = data.get('usage', {})

            print(f"\n✅ 응답 성공 (소요시간: {elapsed_time:.2f}초)")
            print(f"\n답변:\n{answer}")
            print(f"\n📊 통계:")
            print(f"  - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  - Total tokens: {usage.get('total_tokens', 'N/A')}")

            return True
        else:
            print(f"\n❌ 오류: HTTP {response.status_code}")
            print(f"응답: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"\n⏱️ 타임아웃 (60초 초과)")
        return False
    except Exception as e:
        print(f"\n❌ 예외 발생: {str(e)}")
        return False


def main():
    """메인 테스트 실행"""
    print("🚀 LM Studio API 테스트 시작")
    print(f"API 엔드포인트: {LM_STUDIO_URL}")

    tests = [
        {
            "prompt": "안녕하세요. 당신은 제대로 작동하고 있나요?",
            "description": "기본 연결 테스트 (한국어)"
        },
        {
            "prompt": "PDF 파일에서 텍스트를 추출하는 Python 코드 예제를 간단히 보여주세요.",
            "description": "PDF 관련 코딩 질문 (한국어)"
        },
        {
            "prompt": "What Python libraries are commonly used for PDF text extraction?",
            "description": "PDF 관련 질문 (영어)"
        }
    ]

    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n\n{'#'*60}")
        print(f"테스트 {i}/{len(tests)}")
        print(f"{'#'*60}")

        success = test_lm_studio(
            test["prompt"],
            test["description"]
        )
        results.append(success)

        if i < len(tests):
            print("\n⏳ 다음 테스트까지 2초 대기...")
            time.sleep(2)

    # 결과 요약
    print(f"\n\n{'='*60}")
    print("📋 테스트 결과 요약")
    print(f"{'='*60}")
    print(f"총 테스트: {len(results)}개")
    print(f"성공: {sum(results)}개")
    print(f"실패: {len(results) - sum(results)}개")
    print(f"성공률: {sum(results)/len(results)*100:.1f}%")

    if all(results):
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")


if __name__ == "__main__":
    main()
