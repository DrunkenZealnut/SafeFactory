"""고용노동부 기준 최저임금 위반 여부 모의계산 함수 (월급제/일급제 지원)

참고: https://www.moel.go.kr/miniWageMain.do

계산 기준:
- 월급제: (1주 소정근로시간 + 유급시간) × 365 ÷ 84
- 일급제: 1일 실제 근로시간 기준 (8시간 초과 시 1.5배)
- 수습근로자: 3개월 이내 + 계약 1년 이상 + 단순노무직 아닌 경우 → 90% 적용
- 산입 범위: 기본급 + 매월 상여금(전액) + 매월 복리후생비(현금) + 기타 수당
- 제외 항목: 연장·야간·휴일 가산임금, 연차수당, 유급휴일 임금(주휴 제외)
"""

import logging
from datetime import date

# ---------------------------------------------------------------------------
# 연도별 최저임금 테이블 (시간급, 원)
# ---------------------------------------------------------------------------
MINIMUM_WAGE_TABLE = {
    2026: 10_320,
    2025: 10_030,
    2024: 9_860,
    2023: 9_620,
    2022: 9_160,
    2021: 8_720,
    2020: 8_590,
    2019: 8_350,
    2018: 7_530,
    2017: 6_470,
}

# 현재 적용 연도 — 최저임금은 법정 고시 금액이므로 테이블을 수동 업데이트해야 합니다.
# MINIMUM_WAGE_TABLE에 새 연도 항목을 추가하면 CURRENT_YEAR도 함께 갱신하세요.
CURRENT_YEAR = 2026
CURRENT_MIN_HOURLY = MINIMUM_WAGE_TABLE[CURRENT_YEAR]

if date.today().year > CURRENT_YEAR:
    logging.warning(
        "[MinimumWage] CURRENT_YEAR(%d)가 실제 연도(%d)보다 과거입니다. "
        "MINIMUM_WAGE_TABLE 업데이트가 필요합니다.",
        CURRENT_YEAR, date.today().year,
    )


def get_minimum_wage(year: int | None = None) -> int:
    """해당 연도의 최저시급을 반환한다. 없으면 현행 기준."""
    if year is None:
        return CURRENT_MIN_HOURLY
    return MINIMUM_WAGE_TABLE.get(year, CURRENT_MIN_HOURLY)


def calculate_monthly_hours(
    weekly_work_hours: float = 40,
    weekly_paid_holiday_hours: float = 8,
    weekly_agreed_holiday_hours: float = 0,
) -> int:
    """월 소정근로시간 산출.

    공식: (1주 소정근로시간 + 유급시간) × 365 ÷ 84
    ※ 84 = 7일 × 12개월
    """
    weekly_total = weekly_work_hours + weekly_paid_holiday_hours + weekly_agreed_holiday_hours
    return round(weekly_total * 365 / 84)


def calculate_minimum_wage(
    basic_wage: int,
    weekly_work_hours: float = 40,
    weekly_paid_holiday_hours: float = 8,
    weekly_agreed_holiday_hours: float = 0,
    monthly_bonus: int = 0,
    welfare_benefits_cash: int = 0,
    other_allowances: int = 0,
    is_probation: bool = False,
    is_contract_over_1_year: bool = True,
    is_simple_labor: bool = False,
    legal_min_hourly_wage: int | None = None,
    year: int | None = None,
) -> dict:
    """고용노동부 기준 최저임금 위반 여부 모의계산 (월급제).

    Args:
        basic_wage: 기본급 (월액)
        weekly_work_hours: 1주 소정근로시간 (기본 40시간)
        weekly_paid_holiday_hours: 1주 주휴시간 (기본 8시간)
        weekly_agreed_holiday_hours: 1주 약정유급휴일 시간
        monthly_bonus: 매월 지급 상여금 (2024년부터 전액 산입)
        welfare_benefits_cash: 매월 현금 지급 복리후생비 (식대·교통비 등)
        other_allowances: 기타 매월 지급 수당 (최저임금 산입 대상)
        is_probation: 수습 3개월 이내 여부
        is_contract_over_1_year: 근로계약 1년 이상 여부
        is_simple_labor: 단순노무직 종사 여부
        legal_min_hourly_wage: 법정 최저시급 (직접 지정 시)
        year: 적용 연도 (미지정 시 현행 기준)

    Returns:
        계산 결과 딕셔너리
    """
    # 법정 최저시급 결정
    if legal_min_hourly_wage is not None:
        min_hourly = legal_min_hourly_wage
    else:
        min_hourly = get_minimum_wage(year)

    # 1. 월 소정근로시간 산출
    monthly_work_hours = calculate_monthly_hours(
        weekly_work_hours, weekly_paid_holiday_hours, weekly_agreed_holiday_hours
    )

    # 2. 최저임금 산입 임금 합산
    total_included_wage = basic_wage + monthly_bonus + welfare_benefits_cash + other_allowances

    # 3. 환산 시간급
    my_hourly_wage = total_included_wage / monthly_work_hours if monthly_work_hours > 0 else 0

    # 4. 수습근로자 감액 (90%) 적용 여부
    applied_min_wage = min_hourly
    probation_applied = False
    if is_probation and is_contract_over_1_year and not is_simple_labor:
        applied_min_wage = round(min_hourly * 0.9)
        probation_applied = True

    # 5. 위반 여부 판별
    is_violation = my_hourly_wage < applied_min_wage

    # 6. 참고: 법정 최저 월급·일급 (주 40시간 기준)
    standard_monthly_hours = calculate_monthly_hours(40, 8, 0)  # 209시간
    min_monthly_wage = min_hourly * standard_monthly_hours
    min_daily_wage = min_hourly * 8

    return {
        "월_근로시간": monthly_work_hours,
        "최저임금_산입총액": total_included_wage,
        "나의_환산_시급": round(my_hourly_wage, 2),
        "적용_법정_최저시급": applied_min_wage,
        "법정_최저시급": min_hourly,
        "위반_여부": "위반" if is_violation else "정상 (위반 아님)",
        "차액_시급": round(my_hourly_wage - applied_min_wage, 2),
        "수습감액_적용": probation_applied,
        "참고_최저월급": min_monthly_wage,
        "참고_최저일급": min_daily_wage,
        "적용연도": year or CURRENT_YEAR,
    }


def calculate_minimum_wage_daily(
    daily_wage: int,
    daily_work_hours: float = 8,
    overtime_hours: float = 0,
    is_probation: bool = False,
    is_contract_over_1_year: bool = True,
    is_simple_labor: bool = False,
    legal_min_hourly_wage: int | None = None,
    year: int | None = None,
) -> dict:
    """고용노동부 기준 최저임금 위반 여부 모의계산 (일급제).

    일급제 근로자: 1일 실제 근로시간 기준
    - 8시간 이내: 통상 시급으로 계산
    - 8시간 초과: 초과분에 1.5배 적용

    Args:
        daily_wage: 일급 (원)
        daily_work_hours: 1일 소정근로시간 (기본 8시간, 최대 8)
        overtime_hours: 1일 연장근로시간
        is_probation: 수습 3개월 이내 여부
        is_contract_over_1_year: 근로계약 1년 이상 여부
        is_simple_labor: 단순노무직 종사 여부
        legal_min_hourly_wage: 법정 최저시급 (직접 지정 시)
        year: 적용 연도

    Returns:
        계산 결과 딕셔너리
    """
    if legal_min_hourly_wage is not None:
        min_hourly = legal_min_hourly_wage
    else:
        min_hourly = get_minimum_wage(year)

    # 소정근로시간은 최대 8시간, 음수 방어
    base_hours = max(0, min(daily_work_hours, 8))
    overtime_hours = max(0, overtime_hours)

    # 나의 환산 시급: 일급 ÷ (기본시간 + 연장시간 × 1.5)
    total_equivalent_hours = base_hours + (overtime_hours * 1.5)
    my_hourly_wage = daily_wage / total_equivalent_hours if total_equivalent_hours > 0 else 0

    # 수습감액
    applied_min_wage = min_hourly
    probation_applied = False
    if is_probation and is_contract_over_1_year and not is_simple_labor:
        applied_min_wage = round(min_hourly * 0.9)
        probation_applied = True

    # 법정 최저 일급 (기본 8시간 기준)
    min_daily_base = min_hourly * base_hours
    min_daily_overtime = round(min_hourly * 1.5) * overtime_hours
    min_daily_total = round(min_daily_base + min_daily_overtime)

    is_violation = my_hourly_wage < applied_min_wage

    return {
        "기본_근로시간": base_hours,
        "연장_근로시간": overtime_hours,
        "나의_일급": daily_wage,
        "나의_환산_시급": round(my_hourly_wage, 2),
        "적용_법정_최저시급": applied_min_wage,
        "법정_최저시급": min_hourly,
        "법정_최저일급_기본": round(min_daily_base),
        "법정_최저일급_연장포함": min_daily_total,
        "위반_여부": "위반" if is_violation else "정상 (위반 아님)",
        "차액_시급": round(my_hourly_wage - applied_min_wage, 2),
        "수습감액_적용": probation_applied,
        "적용연도": year or CURRENT_YEAR,
    }


if __name__ == "__main__":
    print("=" * 50)
    print(f"  {CURRENT_YEAR}년 최저임금: 시급 {CURRENT_MIN_HOURLY:,}원")
    print(f"  월 환산액 (주 40시간): {CURRENT_MIN_HOURLY * 209:,}원")
    print(f"  일급 (8시간): {CURRENT_MIN_HOURLY * 8:,}원")
    print("=" * 50)

    # 월급제 예시
    print("\n[월급제 모의계산]")
    result = calculate_minimum_wage(
        basic_wage=2_000_000,
        weekly_work_hours=40,
        weekly_paid_holiday_hours=8,
        monthly_bonus=100_000,
        welfare_benefits_cash=100_000,
    )
    for k, v in result.items():
        print(f"  {k}: {v:,}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    # 일급제 예시
    print("\n[일급제 모의계산]")
    result_daily = calculate_minimum_wage_daily(
        daily_wage=90_000,
        daily_work_hours=8,
        overtime_hours=2,
    )
    for k, v in result_daily.items():
        print(f"  {k}: {v:,}" if isinstance(v, (int, float)) else f"  {k}: {v}")
