"""Classify laborlaw questions into calculation / legal / hybrid tracks."""

import re
import logging


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------
_CALC_KEYWORDS = (
    '실수령', '실수령액', '세후', '세전', '수령액', '계산', '얼마',
    '보험료', '공제', '원천징수',
)
_WAGE_KEYWORDS = ('연봉', '월급', '급여', '월소득', '임금', '봉급')
_INSURANCE_KEYWORDS = ('4대보험', '국민연금', '건강보험', '고용보험', '산재보험', '보험료')
_MINIMUM_WAGE_KEYWORDS = ('최저임금', '최저시급', '최저 임금', '최저 시급')
_OVERTIME_KEYWORDS = ('연장근로', '야간근로', '휴일근로', '가산수당', '초과근무', '오버타임')
_WEEKLY_PAY_KEYWORDS = ('주휴수당', '주휴', '주급')
_SEVERANCE_KEYWORDS = ('퇴직금', '퇴직급여')
_ANNUAL_LEAVE_KEYWORDS = ('연차', '연차휴가', '유급휴가', '월차', '연차일수', '연차계산')
_INCOME_TAX_KEYWORDS = ('소득세', '근로소득세', '간이세액', '원천징수세', '지방소득세')

_LEGAL_ONLY_KEYWORDS = (
    '기준', '요건', '조건', '의무', '판례', '판단', '규정',
    '위반', '신고', '절차', '법률', '법조항', '적용',
    '예외', '특례', '제외', '범위',
)

# ---------------------------------------------------------------------------
# Numeric extraction patterns
# ---------------------------------------------------------------------------
_PAT_AMOUNT_MAN = re.compile(
    r'(?:연봉|월급|급여|월소득|임금|봉급|기본급|세전|월[\s]?급여)\s*'
    r'([\d,]+)\s*만\s*원'
)
_PAT_AMOUNT_WON = re.compile(
    r'(?:연봉|월급|급여|월소득|임금|봉급|기본급|세전|월[\s]?급여)\s*'
    r'([\d,]+)\s*원'
)
# 세후/실수령 역산: "세후 300만원", "실수령 300만원", "실수령액 250만원"
_PAT_NET_AMOUNT_MAN = re.compile(
    r'(?:세후|실수령|실수령액|수령액|순수익)\s*([\d,]+)\s*만\s*원'
)
_PAT_NET_AMOUNT_WON = re.compile(
    r'(?:세후|실수령|실수령액|수령액|순수익)\s*([\d,]+)\s*원'
)
_PAT_HOURLY = re.compile(r'시급\s*[은이가도는]?\s*([\d,]+)\s*원?')
_PAT_DAILY_WAGE = re.compile(r'일급\s*([\d,]+)\s*(만\s*)?원')
_PAT_DAILY_HOURS = re.compile(r'(?:하루|1일|일)\s*(\d+)\s*시간')
_PAT_OVERTIME_HOURS = re.compile(r'연장\s*(\d+)\s*시간')
_PAT_HOURS = re.compile(r'(?:주|주당|1주)\s*(\d+)\s*시간')
_PAT_YEARS = re.compile(r'(\d+)\s*(?:년|연)\s*(?:근[무속]|근무|재직|경력)')
_PAT_MONTHS = re.compile(r'(\d+)\s*개월\s*(?:근[무속]|근무|재직|경력)')
_PAT_DEPENDENTS = re.compile(r'부양\s*가족\s*(\d+)')
_PAT_CHILDREN = re.compile(r'(?:자녀|아이)\s*(\d+)')
_PAT_BONUS = re.compile(r'상여금?\s*([\d,]+)\s*(?:만\s*)?원')
_PAT_WELFARE = re.compile(r'(?:식대|교통비|복리후생)\s*([\d,]+)\s*(?:만\s*)?원')
_PAT_SALARY_TYPE_ANNUAL = re.compile(r'연봉')
_PAT_SALARY_TYPE_MONTHLY = re.compile(r'월급|월[\s]?급여|월소득')
_PAT_DATE = re.compile(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})')
# Work schedule parsing
_PAT_TIME_RANGE = re.compile(
    r'(\d{1,2})\s*시\s*(\d{1,2})?\s*분?\s*'
    r'[~에부터까지\-]\s*'
    r'(\d{1,2})\s*시\s*(\d{1,2})?\s*분?'
)
_PAT_BREAK_MIN = re.compile(r'(?:휴식|쉬는\s*시간|휴게)\s*(\d+)\s*분')
_PAT_LUNCH_MIN = re.compile(r'점심\s*(?:시간)?\s*(\d+)\s*분')
_PAT_DOUBLE_BREAK = re.compile(r'오전\s*(?:오후|,\s*오후)\s*(\d+)\s*분')
_PAT_WORK_DAYS = re.compile(r'주\s*(\d)\s*일')
_MONTHLY_WAGE_KEYWORDS = ('월급', '월소득', '월 급여', '한달', '한 달')
_PAT_HIRE_DATE = re.compile(r'입사\s*(?:일|날짜|일자)?\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})')
_PAT_WITHHOLDING = re.compile(r'(?:원천징수|원천)\s*(\d+)\s*%')


def _parse_number(s: str) -> int:
    """Remove commas and convert to int."""
    return int(s.replace(',', ''))


def _has_any(query_lower: str, keywords: tuple) -> bool:
    return any(kw in query_lower for kw in keywords)


def _has_numbers(query: str) -> bool:
    """Check if query contains meaningful numbers (not just 4대보험)."""
    cleaned = re.sub(r'4대', '', query)
    return bool(re.search(r'\d{2,}', cleaned))


def classify_labor_question(query: str) -> dict:
    """Classify a laborlaw question into type and extract parameters.

    Returns:
        {
            'type': 'calculation' | 'legal' | 'hybrid',
            'calc_type': 'wage' | 'insurance' | 'minimum_wage' | 'overtime'
                         | 'weekly_holiday' | 'severance' | 'annual_leave'
                         | 'income_tax' | None,
            'params': dict,
            'search_needed': bool,
        }
    """
    q = query.strip()
    ql = q.lower()
    params = {}

    # --- Extract numeric parameters ---
    # Net amount (세후/실수령 N만원 → 역산)
    m = _PAT_NET_AMOUNT_MAN.search(q)
    if m:
        params['net_amount'] = _parse_number(m.group(1)) * 10000
        params['direction'] = 'net_to_gross'
        params['salary_type'] = '연봉' if _PAT_SALARY_TYPE_ANNUAL.search(q) else '월급'
    if 'net_amount' not in params:
        m = _PAT_NET_AMOUNT_WON.search(q)
        if m:
            params['net_amount'] = _parse_number(m.group(1))
            params['direction'] = 'net_to_gross'
            params['salary_type'] = '연봉' if _PAT_SALARY_TYPE_ANNUAL.search(q) else '월급'

    # Salary (만원)
    m = _PAT_AMOUNT_MAN.search(q)
    if m:
        params['amount'] = _parse_number(m.group(1)) * 10000
        if 'salary_type' not in params:
            params['salary_type'] = '연봉' if _PAT_SALARY_TYPE_ANNUAL.search(q) else '월급'

    # Salary (원)
    if 'amount' not in params:
        m = _PAT_AMOUNT_WON.search(q)
        if m:
            params['amount'] = _parse_number(m.group(1))
            if 'salary_type' not in params:
                params['salary_type'] = '연봉' if _PAT_SALARY_TYPE_ANNUAL.search(q) else '월급'

    # Hourly wage
    m = _PAT_HOURLY.search(q)
    if m:
        params['hourly_wage'] = _parse_number(m.group(1))

    # Work schedule → daily actual hours → weekly hours
    m_time = _PAT_TIME_RANGE.search(q)
    if m_time:
        start_h, start_m = int(m_time.group(1)), int(m_time.group(2) or 0)
        end_h, end_m = int(m_time.group(3)), int(m_time.group(4) or 0)
        total_min = (end_h * 60 + end_m) - (start_h * 60 + start_m)
        # 야간 근무: 종료 시간이 시작 시간보다 이른 경우 24시간 추가
        if total_min <= 0:
            total_min += 24 * 60

        # 휴식시간 파싱
        break_total = 0
        # "오전 오후 N분" 패턴 → 2회 처리
        m_double = _PAT_DOUBLE_BREAK.search(q)
        if m_double:
            break_total += int(m_double.group(1)) * 2
        else:
            for bm in _PAT_BREAK_MIN.finditer(q):
                break_total += int(bm.group(1))
        m_lunch = _PAT_LUNCH_MIN.search(q)
        if m_lunch:
            break_total += int(m_lunch.group(1))

        actual_min = total_min - break_total
        params['daily_work_minutes'] = actual_min
        params['daily_break_minutes'] = break_total
        params['schedule_start'] = f"{start_h:02d}:{start_m:02d}"
        params['schedule_end'] = f"{end_h:02d}:{end_m:02d}"

        # 주당 근무일수 (기본 5일)
        m_days = _PAT_WORK_DAYS.search(q)
        work_days = int(m_days.group(1)) if m_days else 5
        params['work_days_per_week'] = work_days

        if 'weekly_hours' not in params:
            params['weekly_hours'] = round(actual_min / 60 * work_days, 2)

    # Daily wage
    m = _PAT_DAILY_WAGE.search(q)
    if m:
        val = _parse_number(m.group(1))
        params['daily_wage'] = val * 10000 if m.group(2) else val
        params['wage_type'] = 'daily'

    # Daily work hours
    m = _PAT_DAILY_HOURS.search(q)
    if m:
        params['daily_hours'] = int(m.group(1))

    # Overtime hours
    m = _PAT_OVERTIME_HOURS.search(q)
    if m:
        params['overtime_hours'] = int(m.group(1))

    # Weekly hours
    m = _PAT_HOURS.search(q)
    if m:
        params['weekly_hours'] = int(m.group(1))

    # Tenure
    m = _PAT_YEARS.search(q)
    if m:
        params['tenure_years'] = int(m.group(1))
    m = _PAT_MONTHS.search(q)
    if m:
        params['tenure_months'] = int(m.group(1))

    # Dependents / children
    m = _PAT_DEPENDENTS.search(q)
    if m:
        params['dependents'] = int(m.group(1))
    m = _PAT_CHILDREN.search(q)
    if m:
        params['children'] = int(m.group(1))

    # Bonus / welfare
    m = _PAT_BONUS.search(q)
    if m:
        val = _parse_number(m.group(1))
        params['monthly_bonus'] = val * 10000 if '만' in q[m.start():m.end()+3] else val
    m = _PAT_WELFARE.search(q)
    if m:
        val = _parse_number(m.group(1))
        params['welfare_cash'] = val * 10000 if '만' in q[m.start():m.end()+3] else val

    # Hire date (for annual leave)
    m = _PAT_HIRE_DATE.search(q)
    if m:
        params['hire_date'] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    elif _has_any(ql, _ANNUAL_LEAVE_KEYWORDS):
        dates = _PAT_DATE.findall(q)
        if dates:
            params['hire_date'] = f"{dates[0][0]}-{int(dates[0][1]):02d}-{int(dates[0][2]):02d}"
            if len(dates) > 1:
                params['end_date'] = f"{dates[1][0]}-{int(dates[1][1]):02d}-{int(dates[1][2]):02d}"

    # Withholding rate (for income tax)
    m = _PAT_WITHHOLDING.search(q)
    if m:
        params['withholding_rate'] = int(m.group(1))

    # Non-taxable income alias
    if params.get('welfare_cash') and _has_any(ql, _INCOME_TAX_KEYWORDS):
        params['non_taxable'] = params['welfare_cash']

    # --- Determine calc_type ---
    calc_type = None
    has_numbers = bool(
        params.get('amount') or params.get('hourly_wage') or
        params.get('daily_wage') or params.get('tenure_years')
    )
    has_net_amount = bool(params.get('net_amount'))

    # 세후 역산 (net_to_gross): 가장 우선 처리
    if has_net_amount and params.get('direction') == 'net_to_gross':
        calc_type = 'wage_reverse'

    has_daily_wage = bool(params.get('daily_wage'))

    # Priority: wage_reverse already set → skip; otherwise domain keywords first
    if calc_type != 'wage_reverse':
        if _has_any(ql, _ANNUAL_LEAVE_KEYWORDS) and (params.get('hire_date') or params.get('tenure_years')):
            calc_type = 'annual_leave'
        elif _has_any(ql, _INCOME_TAX_KEYWORDS) and has_numbers:
            calc_type = 'income_tax'
        elif _has_any(ql, _MINIMUM_WAGE_KEYWORDS) and (has_numbers or has_daily_wage):
            calc_type = 'minimum_wage'
        elif _has_any(ql, _SEVERANCE_KEYWORDS) and has_numbers:
            calc_type = 'severance'
        elif _has_any(ql, _INSURANCE_KEYWORDS) and has_numbers:
            calc_type = 'insurance'
        elif params.get('hourly_wage') and params.get('weekly_hours') and _has_any(ql, _MONTHLY_WAGE_KEYWORDS + _CALC_KEYWORDS):
            calc_type = 'monthly_wage'
        elif params.get('hourly_wage') and params.get('daily_work_minutes'):
            calc_type = 'monthly_wage'
        elif _has_any(ql, _WEEKLY_PAY_KEYWORDS) and (params.get('hourly_wage') or params.get('weekly_hours')):
            calc_type = 'weekly_holiday'
        elif _has_any(ql, _OVERTIME_KEYWORDS) and has_numbers:
            calc_type = 'overtime'
        elif _has_any(ql, _WAGE_KEYWORDS) and has_numbers and _has_any(ql, _CALC_KEYWORDS):
            calc_type = 'wage'
        elif has_numbers and _has_any(ql, _CALC_KEYWORDS):
            calc_type = 'wage'

    # --- Determine question type ---
    has_legal_keywords = _has_any(ql, _LEGAL_ONLY_KEYWORDS)

    if calc_type and has_legal_keywords:
        q_type = 'hybrid'
    elif calc_type:
        q_type = 'calculation'
    else:
        q_type = 'legal'

    # All tracks still search documents (for citations / legal basis)
    search_needed = True

    logging.info("[LaborClassifier] type=%s, calc_type=%s, params=%s", q_type, calc_type, params)

    return {
        'type': q_type,
        'calc_type': calc_type,
        'params': params,
        'search_needed': search_needed,
    }
