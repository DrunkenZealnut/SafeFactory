"""Execute labor-law calculations and format results for LLM prompt injection."""

import logging

from calculator.wage_calculator import WageCalculator
from calculator.insurance_calculator import InsuranceCalculator, CompanySize
from calculator.minimum_wage import (
    calculate_minimum_wage,
    calculate_minimum_wage_daily,
    CURRENT_MIN_HOURLY,
    CURRENT_YEAR,
)
from calculator.retirement_calculator import RetirementPayCalculator
from calculator.annual_leave_calculator import AnnualLeaveCalculator
from calculator.income_tax_calculator import IncomeTaxCalculator


TAX_FREE_CAP = 200_000  # 비과세 상한액 (원/월)


def _fmt(n) -> str:
    """Format a number as Korean currency string."""
    if isinstance(n, float):
        return f"{n:,.0f}"
    return f"{n:,}"


def _cap_tax_free(welfare_cash: int) -> tuple[int, int, bool]:
    """Apply tax-free cap and return (original, capped, was_adjusted)."""
    capped = min(welfare_cash, TAX_FREE_CAP)
    return welfare_cash, capped, welfare_cash > TAX_FREE_CAP


def run_labor_calculation(classification: dict) -> dict | None:
    """Run the appropriate calculator based on classifier output.

    Args:
        classification: Output from classify_labor_question().

    Returns:
        {
            'calc_type': str,
            'input_summary': str,
            'result': dict,       # raw calculator output
            'formatted': str,     # markdown text for LLM prompt
        }
        or None if calculation is not applicable or fails.
    """
    calc_type = classification.get('calc_type')
    params = classification.get('params', {})

    if not calc_type:
        return None

    try:
        if calc_type == 'wage':
            return _run_wage(params)
        elif calc_type == 'wage_reverse':
            return _run_wage_reverse(params)
        elif calc_type == 'insurance':
            return _run_insurance(params)
        elif calc_type == 'minimum_wage':
            return _run_minimum_wage(params)
        elif calc_type == 'overtime':
            return _run_overtime(params)
        elif calc_type == 'weekly_holiday':
            return _run_weekly_holiday(params)
        elif calc_type == 'severance':
            return _run_severance(params)
        elif calc_type == 'annual_leave':
            return _run_annual_leave(params)
        elif calc_type == 'income_tax':
            return _run_income_tax(params)
    except Exception as e:
        logging.warning(
            "[LaborCalculator] %s calculation failed: %s",
            calc_type, e, exc_info=True
        )
    return None


# ---------------------------------------------------------------------------
# Wage (실수령액) calculation
# ---------------------------------------------------------------------------
def _run_wage(params: dict) -> dict | None:
    amount = params.get('amount')
    if not amount or amount <= 0:
        return None

    salary_type = params.get('salary_type', '연봉')
    dependents = params.get('dependents', 1)
    children = params.get('children', 0)
    tax_free_original, tax_free, tax_free_adjusted = _cap_tax_free(params.get('welfare_cash', 0))

    calc = WageCalculator()
    if salary_type == '연봉':
        result = calc.calculate_from_annual(
            annual_salary=amount,
            tax_free_monthly=tax_free,
            dependents=dependents,
            children_8_to_20=children,
        )
        input_summary = f"연봉 {_fmt(amount)}원"
    else:
        result = calc.calculate_from_monthly(
            monthly_salary=amount,
            tax_free_monthly=tax_free,
            dependents=dependents,
            children_8_to_20=children,
        )
        input_summary = f"월급 {_fmt(amount)}원"

    if dependents > 1:
        input_summary += f", 부양가족 {dependents}인"
    if children > 0:
        input_summary += f", 자녀 {children}명"
    if tax_free > 0:
        input_summary += f", 비과세 {_fmt(tax_free)}원"
    if tax_free_adjusted:
        input_summary += f" (입력 {_fmt(tax_free_original)}원 → 상한 {_fmt(TAX_FREE_CAP)}원 적용)"

    ded = result['근로자_공제내역']
    net = result['실수령액']

    formatted = f"""### 급여 계산 결과 ({input_summary})

| 항목 | 금액 |
|------|------|
| 월 급여 | {_fmt(result['입력정보']['월급여'])}원 |
| 국민연금 | -{_fmt(ded['국민연금'])}원 |
| 건강보험 | -{_fmt(ded['건강보험'])}원 |
| 장기요양보험 | -{_fmt(ded['장기요양보험'])}원 |
| 고용보험 | -{_fmt(ded['고용보험'])}원 |
| 소득세 | -{_fmt(ded['소득세'])}원 |
| 지방소득세 | -{_fmt(ded['지방소득세'])}원 |
| **공제합계** | **-{_fmt(ded['공제합계'])}원** |
| **월 실수령액** | **{_fmt(net['월_실수령액'])}원** |
| 연 실수령액(추정) | {_fmt(net['연_실수령액_추정'])}원 |"""

    return {
        'calc_type': 'wage',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Wage reverse (세후 → 세전 역산) calculation
# ---------------------------------------------------------------------------
def _run_wage_reverse(params: dict) -> dict | None:
    net_amount = params.get('net_amount')
    if not net_amount or net_amount <= 0:
        return None

    salary_type = params.get('salary_type', '월급')
    dependents = params.get('dependents', 1)
    children = params.get('children', 0)
    tax_free_original, tax_free, tax_free_adjusted = _cap_tax_free(params.get('welfare_cash', 0))

    # 연봉으로 입력된 경우 월 기준으로 변환
    target_monthly_net = net_amount // 12 if salary_type == '연봉' else net_amount

    calc = WageCalculator()
    result = calc.calculate_from_net(
        target_net_monthly=target_monthly_net,
        tax_free_monthly=tax_free,
        dependents=dependents,
        children_8_to_20=children,
    )

    rev = result['역산정보']
    ded = result['근로자_공제내역']

    if salary_type == '연봉':
        input_summary = f"희망 세후 연봉 {_fmt(net_amount)}원 (월 {_fmt(target_monthly_net)}원)"
    else:
        input_summary = f"희망 세후 월급 {_fmt(net_amount)}원"

    if dependents > 1:
        input_summary += f", 부양가족 {dependents}인"
    if children > 0:
        input_summary += f", 자녀 {children}명"
    if tax_free > 0:
        input_summary += f", 비과세 {_fmt(tax_free)}원"
    if tax_free_adjusted:
        input_summary += f" (입력 {_fmt(tax_free_original)}원 → 상한 {_fmt(TAX_FREE_CAP)}원 적용)"

    formatted = f"""### 세후 → 세전 역산 결과 ({input_summary})

| 항목 | 금액 |
|------|------|
| 희망 월 실수령액 | {_fmt(target_monthly_net)}원 |
| **필요 세전 월급** | **{_fmt(rev['필요_세전_월급'])}원** |
| **필요 세전 연봉** | **{_fmt(rev['필요_세전_연봉'])}원** |
| 국민연금 | -{_fmt(ded['국민연금'])}원 |
| 건강보험 | -{_fmt(ded['건강보험'])}원 |
| 장기요양보험 | -{_fmt(ded['장기요양보험'])}원 |
| 고용보험 | -{_fmt(ded['고용보험'])}원 |
| 소득세 | -{_fmt(ded['소득세'])}원 |
| 지방소득세 | -{_fmt(ded['지방소득세'])}원 |
| 공제합계 | -{_fmt(ded['공제합계'])}원 |
| 실제 월 실수령액 | {_fmt(rev['실제_실수령액'])}원 |"""

    return {
        'calc_type': 'wage_reverse',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Insurance (4대보험) calculation
# ---------------------------------------------------------------------------
def _run_insurance(params: dict) -> dict | None:
    amount = params.get('amount')
    if not amount or amount <= 0:
        return None

    salary_type = params.get('salary_type', '월급')
    monthly = amount // 12 if salary_type == '연봉' else amount
    tax_free_original, tax_free, tax_free_adjusted = _cap_tax_free(params.get('welfare_cash', 0))

    calc = InsuranceCalculator()
    result = calc.calculate_all(
        monthly_income=monthly,
        non_taxable=tax_free,
        company_size=CompanySize.UNDER_150,
    )

    input_summary = f"월 소득 {_fmt(monthly)}원"
    if tax_free > 0:
        input_summary += f", 비과세 {_fmt(tax_free)}원"
    if tax_free_adjusted:
        input_summary += f" (입력 {_fmt(tax_free_original)}원 → 상한 {_fmt(TAX_FREE_CAP)}원 적용)"

    s = result['합계']
    formatted = f"""### 4대보험료 계산 결과 ({input_summary})

| 보험 | 근로자 | 사업주 |
|------|--------|--------|
| 국민연금 | {_fmt(result['국민연금']['근로자부담'])}원 | {_fmt(result['국민연금']['사업주부담'])}원 |
| 건강보험 | {_fmt(result['건강보험']['근로자부담'])}원 | {_fmt(result['건강보험']['사업주부담'])}원 |
| 장기요양보험 | {_fmt(result['장기요양보험']['근로자부담'])}원 | {_fmt(result['장기요양보험']['사업주부담'])}원 |
| 고용보험 | {_fmt(result['고용보험']['근로자부담'])}원 | {_fmt(result['고용보험']['사업주부담'])}원 |
| 산재보험 | - | {_fmt(result['산재보험']['사업주부담'])}원 |
| **합계** | **{_fmt(s['근로자부담_합계'])}원** | **{_fmt(s['사업주부담_합계'])}원** |"""

    return {
        'calc_type': 'insurance',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Minimum wage (최저임금 위반 여부) calculation
# ---------------------------------------------------------------------------
def _run_minimum_wage(params: dict) -> dict | None:
    amount = params.get('amount')
    hourly = params.get('hourly_wage')
    daily_wage = params.get('daily_wage')
    wage_type = params.get('wage_type', 'monthly')  # monthly | daily

    if not amount and not hourly and not daily_wage:
        return None

    weekly_hours = params.get('weekly_hours', 40)
    bonus = params.get('monthly_bonus', 0)
    welfare = params.get('welfare_cash', 0)
    overtime_hours = params.get('overtime_hours', 0)

    legal_min = CURRENT_MIN_HOURLY

    # --- 일급제 계산 ---
    if wage_type == 'daily' or daily_wage:
        dw = daily_wage or amount
        if not dw:
            return None
        daily_hours = params.get('daily_hours', 8)
        result = calculate_minimum_wage_daily(
            daily_wage=dw,
            daily_work_hours=daily_hours,
            overtime_hours=overtime_hours,
        )
        input_summary = f"일급 {_fmt(dw)}원, {daily_hours}시간"
        if overtime_hours > 0:
            input_summary += f" + 연장 {overtime_hours}시간"

        formatted = f"""### 최저임금 위반 여부 계산 - 일급제 ({input_summary})

| 항목 | 값 |
|------|-----|
| 기본 근로시간 | {result['기본_근로시간']}시간 |
| 연장 근로시간 | {result['연장_근로시간']}시간 |
| 나의 일급 | {_fmt(result['나의_일급'])}원 |
| 나의 환산 시급 | {_fmt(result['나의_환산_시급'])}원 |
| {CURRENT_YEAR}년 법정 최저시급 | {_fmt(result['법정_최저시급'])}원 |
| 법정 최저일급 (기본) | {_fmt(result['법정_최저일급_기본'])}원 |
| 법정 최저일급 (연장포함) | {_fmt(result['법정_최저일급_연장포함'])}원 |
| **판정** | **{result['위반_여부']}** |
| 차액(시급) | {_fmt(result['차액_시급'])}원 |"""

        return {
            'calc_type': 'minimum_wage',
            'input_summary': input_summary,
            'result': result,
            'formatted': formatted,
        }

    # --- 월급제 계산 ---
    if amount:
        salary_type = params.get('salary_type', '월급')
        basic_wage = amount // 12 if salary_type == '연봉' else amount
    elif hourly:
        basic_wage = int(hourly * weekly_hours * 4.345)
    else:
        return None

    result = calculate_minimum_wage(
        basic_wage=basic_wage,
        weekly_work_hours=weekly_hours,
        monthly_bonus=bonus,
        welfare_benefits_cash=welfare,
    )

    input_summary = f"기본급 {_fmt(basic_wage)}원/월"
    if bonus > 0:
        input_summary += f", 상여금 {_fmt(bonus)}원"
    if welfare > 0:
        input_summary += f", 복리후생비 {_fmt(welfare)}원"

    formatted = f"""### 최저임금 위반 여부 계산 - 월급제 ({input_summary})

| 항목 | 값 |
|------|-----|
| 월 소정근로시간 | {result['월_근로시간']}시간 |
| 최저임금 산입 총액 | {_fmt(result['최저임금_산입총액'])}원 |
| 나의 환산 시급 | {_fmt(result['나의_환산_시급'])}원 |
| {CURRENT_YEAR}년 법정 최저시급 | {_fmt(result['법정_최저시급'])}원 |
| 적용 최저시급 | {_fmt(result['적용_법정_최저시급'])}원 |
| **판정** | **{result['위반_여부']}** |
| 차액(시급) | {_fmt(result['차액_시급'])}원 |
| 참고: 최저 월급 (주40h) | {_fmt(result['참고_최저월급'])}원 |
| 참고: 최저 일급 (8h) | {_fmt(result['참고_최저일급'])}원 |"""

    return {
        'calc_type': 'minimum_wage',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Overtime pay (가산수당) — formula only, no separate calculator module
# ---------------------------------------------------------------------------
def _run_overtime(params: dict) -> dict | None:
    amount = params.get('amount')
    hourly = params.get('hourly_wage')
    if not amount and not hourly:
        return None

    if hourly:
        base_hourly = hourly
    elif amount:
        salary_type = params.get('salary_type', '월급')
        monthly = amount // 12 if salary_type == '연봉' else amount
        base_hourly = round(monthly / 209)
    else:
        return None

    overtime_50 = round(base_hourly * 1.5)
    overtime_100 = round(base_hourly * 2.0)
    night_50 = round(base_hourly * 1.5)

    input_summary = f"통상시급 {_fmt(base_hourly)}원"

    formatted = f"""### 가산수당 계산 ({input_summary})

| 근로 유형 | 가산율 | 시급 |
|-----------|--------|------|
| 통상근로 | 100% | {_fmt(base_hourly)}원 |
| 연장근로 (8h 초과) | 150% | {_fmt(overtime_50)}원 |
| 야간근로 (22~06시) | 150% | {_fmt(night_50)}원 |
| 휴일근로 (8h 이내) | 150% | {_fmt(overtime_50)}원 |
| 휴일근로 (8h 초과) | 200% | {_fmt(overtime_100)}원 |"""

    return {
        'calc_type': 'overtime',
        'input_summary': input_summary,
        'result': {
            'base_hourly': base_hourly,
            'overtime_150': overtime_50,
            'overtime_200': overtime_100,
        },
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Weekly holiday pay (주휴수당) calculation
# ---------------------------------------------------------------------------
def _run_weekly_holiday(params: dict) -> dict | None:
    hourly = params.get('hourly_wage')
    weekly_hours = params.get('weekly_hours')
    if not hourly and not weekly_hours:
        return None

    if not hourly:
        hourly = CURRENT_MIN_HOURLY
    if not weekly_hours:
        weekly_hours = 40

    # 주휴시간 = (주당 근무시간 / 40) × 8, 상한 8시간
    weekly_holiday_hours = min((weekly_hours / 40) * 8, 8)
    weekly_holiday_pay = round(hourly * weekly_holiday_hours)
    monthly_holiday_pay = round(weekly_holiday_pay * 4.345)
    monthly_total = round(hourly * (weekly_hours + weekly_holiday_hours) * 4.345)

    input_summary = f"시급 {_fmt(hourly)}원, 주 {weekly_hours}시간"

    formatted = f"""### 주휴수당 계산 ({input_summary})

| 항목 | 값 |
|------|-----|
| 주휴시간 | {weekly_holiday_hours:.1f}시간 |
| 주휴수당 (주) | {_fmt(weekly_holiday_pay)}원 |
| 주휴수당 (월) | {_fmt(monthly_holiday_pay)}원 |
| **월 총급여 (주휴 포함)** | **{_fmt(monthly_total)}원** |"""

    return {
        'calc_type': 'weekly_holiday',
        'input_summary': input_summary,
        'result': {
            'weekly_holiday_hours': weekly_holiday_hours,
            'weekly_holiday_pay': weekly_holiday_pay,
            'monthly_holiday_pay': monthly_holiday_pay,
            'monthly_total': monthly_total,
        },
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Severance pay (퇴직금) calculation — 고용노동부 공식 기준
# ---------------------------------------------------------------------------
def _run_severance(params: dict) -> dict | None:
    # ----- 날짜 기반 계산 (상세 모드) -----
    start_date = params.get('start_date')
    end_date = params.get('end_date')

    if start_date and end_date:
        basic = params.get('monthly_basic_pay')
        other = params.get('monthly_other_pay', 0)
        annual_bonus = params.get('annual_bonus', 0)
        annual_leave_pay = params.get('annual_leave_pay', 0)
        excluded_avg = params.get('excluded_days_avg', 0)
        excluded_svc = params.get('excluded_days_service', 0)
        ordinary = params.get('ordinary_daily_wage')

        if not basic:
            return None

        try:
            calc = RetirementPayCalculator(
                start_date=start_date,
                end_date=end_date,
                monthly_basic_pay=basic,
                monthly_other_pay=other,
                annual_bonus=annual_bonus,
                annual_leave_pay=annual_leave_pay,
                excluded_days_avg=excluded_avg,
                excluded_days_service=excluded_svc,
                ordinary_daily_wage=ordinary,
            )
            result = calc.calculate()
        except ValueError as e:
            return {
                'calc_type': 'severance',
                'input_summary': str(e),
                'result': {'error': str(e)},
                'formatted': f"### 퇴직금 계산 오류\n\n{e}",
            }

        info = result['입력정보']
        avg = result['평균임금_산정']
        sev = result['퇴직금_산출']

        input_summary = (
            f"{info['입사일자']} ~ {info['퇴직일자']}, "
            f"재직 {info['재직일수']}일"
        )

        wage_note = ""
        if sev['통상임금_적용여부']:
            wage_note = "\n| ⚠️ 통상임금 적용 | 1일 통상임금이 평균임금보다 큼 |"

        formatted = f"""### 퇴직금 계산 결과 ({input_summary})

**평균임금 산정**

| 항목 | 금액 |
|------|------|
| 3개월 임금총액 (기본급+기타수당) | {_fmt(avg['임금총액_3개월'])}원 |
| 상여금 가산액 (연 {_fmt(calc.annual_bonus)}원 × 3/12) | {_fmt(avg['상여금_가산액'])}원 |
| 연차수당 가산액 (연 {_fmt(calc.annual_leave_pay)}원 × 3/12) | {_fmt(avg['연차수당_가산액'])}원 |
| 평균임금 기초금액 합계 | {_fmt(avg['평균임금_기초금액'])}원 |
| 퇴직 전 3개월 산정일수 | {avg['산정기간_일수']}일 |
| **1일 평균임금** | **{_fmt(avg['1일_평균임금'])}원** |{wage_note}

**퇴직금 산출**

| 항목 | 값 |
|------|-----|
| 적용 1일 임금 | {_fmt(sev['적용_1일_임금'])}원 |
| 재직일수 | {sev['재직일수']}일 |
| **퇴직금** (1일임금 × 30 × 재직일수/365) | **{_fmt(sev['퇴직금'])}원** |"""

        return {
            'calc_type': 'severance',
            'input_summary': input_summary,
            'result': result,
            'formatted': formatted,
        }

    # ----- 간이 계산 (월급/연봉 + 근속기간만 입력) -----
    amount = params.get('amount')
    if not amount or amount <= 0:
        return None

    salary_type = params.get('salary_type', '월급')
    monthly = amount // 12 if salary_type == '연봉' else amount

    tenure_years = params.get('tenure_years', 0)
    tenure_months = params.get('tenure_months', 0)
    total_months = tenure_years * 12 + tenure_months
    if total_months < 12:
        total_months = 12

    daily_wage = monthly / 30
    total_days = round(total_months * 30.42)
    severance = round(daily_wage * 30 * (total_days / 365))

    input_summary = f"월급 {_fmt(monthly)}원, 근속 {total_months}개월"

    formatted = f"""### 퇴직금 계산 ({input_summary})

| 항목 | 값 |
|------|-----|
| 월 평균임금 | {_fmt(monthly)}원 |
| 1일 평균임금 | {_fmt(round(daily_wage))}원 |
| 근속기간 | {total_months}개월 ({total_days}일) |
| **퇴직금** | **{_fmt(severance)}원** |

> 💡 입사일/퇴직일, 상여금, 연차수당을 입력하면 더 정확한 계산이 가능합니다."""

    return {
        'calc_type': 'severance',
        'input_summary': input_summary,
        'result': {
            'monthly_wage': monthly,
            'daily_wage': round(daily_wage),
            'total_days': total_days,
            'severance': severance,
        },
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Annual leave (연차유급휴가) calculation — 근로기준법 제60조
# ---------------------------------------------------------------------------
def _run_annual_leave(params: dict) -> dict | None:
    hire_date = params.get('hire_date') or params.get('start_date')
    if not hire_date:
        return None

    end_date = params.get('end_date') or params.get('resignation_date')

    try:
        calc = AnnualLeaveCalculator(
            hire_date=hire_date,
            end_date=end_date,
        )
        result = calc.calculate()
    except ValueError as e:
        return {
            'calc_type': 'annual_leave',
            'input_summary': str(e),
            'result': {'error': str(e)},
            'formatted': f"### 연차휴가 계산 오류\n\n{e}",
        }

    info = result['입력정보']
    yearly = result['연도별_내역']
    total = result['총_발생_연차일수']

    input_summary = (
        f"입사 {info['입사일자']}, 기준 {info['기준일자']}, "
        f"만 {info['만_근속연수']}년"
    )

    # 연도별 테이블 생성
    rows = []
    for y in yearly:
        rows.append(
            f"| {y['근무년차']}년차 | {y['기간_시작']} ~ {y['기간_종료']} "
            f"| {y['유형']} | **{y['발생일수']}일** | {y['비고']} |"
        )
    table_body = '\n'.join(rows)

    formatted = f"""### 연차유급휴가 계산 결과 ({input_summary})

| 근무년차 | 기간 | 유형 | 발생일수 | 비고 |
|----------|------|------|----------|------|
{table_body}
| **합계** | | | **{total}일** | |"""

    return {
        'calc_type': 'annual_leave',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }


# ---------------------------------------------------------------------------
# Income tax (근로소득세) calculation — 간이세액표 산출 공식 기준
# ---------------------------------------------------------------------------
def _run_income_tax(params: dict) -> dict | None:
    amount = params.get('amount')
    if not amount or amount <= 0:
        return None

    salary_type = params.get('salary_type', '월급')
    monthly = amount // 12 if salary_type == '연봉' else amount

    non_taxable = params.get('non_taxable', 0)
    if non_taxable is None:
        non_taxable = 0
    dependents = params.get('dependents', 1)
    children = params.get('children', 0)
    withholding_rate = params.get('withholding_rate', 100)

    calc = IncomeTaxCalculator(
        monthly_salary=monthly,
        non_taxable=non_taxable,
        dependents=dependents,
        children_8_to_20=children,
        withholding_rate=withholding_rate,
    )
    result = calc.calculate()

    tax = result['최종_세액']
    detail = result['소득공제_내역']
    tax_calc = result['세액_산출']

    input_summary = f"월급 {_fmt(monthly)}원"
    if non_taxable > 0:
        input_summary += f", 비과세 {_fmt(non_taxable)}원"
    if dependents > 1:
        input_summary += f", 부양가족 {dependents}인"
    if children > 0:
        input_summary += f", 자녀 {children}명"
    if withholding_rate != 100:
        input_summary += f", 원천징수 {withholding_rate}%"

    formatted = f"""### 근로소득세 계산 결과 ({input_summary})

**소득공제 내역 (연간)**

| 항목 | 금액 |
|------|------|
| 연간 총급여 | {_fmt(detail['연간_총급여'])}원 |
| 근로소득공제 | -{_fmt(detail['근로소득공제'])}원 |
| 근로소득금액 | {_fmt(detail['근로소득금액'])}원 |
| 인적공제 | -{_fmt(detail['인적공제'])}원 |
| 국민연금 공제 | -{_fmt(detail['국민연금_공제'])}원 |
| 건강보험 공제 | -{_fmt(detail['건강보험_공제'])}원 |
| 장기요양보험 공제 | -{_fmt(detail['장기요양보험_공제'])}원 |
| 고용보험 공제 | -{_fmt(detail['고용보험_공제'])}원 |
| **과세표준** | **{_fmt(detail['과세표준'])}원** |

**세액 산출**

| 항목 | 금액 |
|------|------|
| 산출세액 | {_fmt(tax_calc['산출세액'])}원/년 |
| 근로소득세액공제 | -{_fmt(tax_calc['근로소득세액공제'])}원 |
| 표준세액공제 | -{_fmt(tax_calc['표준세액공제'])}원 |
| 연간 결정세액 | {_fmt(tax_calc['연간_결정세액'])}원 |

**월 납부 세액**

| 항목 | 금액 |
|------|------|
| 근로소득세 | {_fmt(tax['근로소득세'])}원 |
| 지방소득세 | {_fmt(tax['지방소득세'])}원 |
| **합계** | **{_fmt(tax['합계'])}원** |"""

    return {
        'calc_type': 'income_tax',
        'input_summary': input_summary,
        'result': result,
        'formatted': formatted,
    }
