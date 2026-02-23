"""근로소득세 계산기 - 국세청 간이세액표 산출 공식 기반.

Reference: https://www.nodong.kr/income_tax

소득세법 제47조 (근로소득공제), 제55조 (세율),
소득세법 제59조 (근로소득세액공제), 시행령 별표2 (간이세액표)

2024.3 ~ 2026.2 간이세액표 기준.
"""


# ---------------------------------------------------------------------------
# 근로소득공제 구간 (소득세법 제47조, 연간 기준)
# (상한, 초과분 세율, 누적 공제액)
# ---------------------------------------------------------------------------
_EARNED_INCOME_DEDUCTION = [
    (5_000_000,   0.70,          0),
    (15_000_000,  0.40,  3_500_000),
    (45_000_000,  0.15,  7_500_000),
    (100_000_000, 0.05, 12_000_000),
    (float('inf'), 0.02, 14_750_000),
]

# ---------------------------------------------------------------------------
# 종합소득세율 (소득세법 제55조, 연간 기준)
# (상한, 세율, 누진공제액)
# ---------------------------------------------------------------------------
_TAX_BRACKETS = [
    (14_000_000,   0.06,          0),
    (50_000_000,   0.15,  1_260_000),
    (88_000_000,   0.24,  5_760_000),
    (150_000_000,  0.35, 15_440_000),
    (300_000_000,  0.38, 19_940_000),
    (500_000_000,  0.40, 25_940_000),
    (1_000_000_000, 0.42, 35_940_000),
    (float('inf'), 0.45, 65_940_000),
]

# ---------------------------------------------------------------------------
# 자녀세액공제 (2024.3~2026.2 간이세액표 기준, 월 기준)
# ---------------------------------------------------------------------------
_CHILD_CREDIT = {1: 12_500, 2: 29_160}
_CHILD_CREDIT_EXTRA_PER = 25_000  # 3명 이상 시 추가 1인당

# ---------------------------------------------------------------------------
# 보험요율 (간이세액표 산출용, 2026 기준)
# ---------------------------------------------------------------------------
_NP_RATE = 0.0475       # 국민연금 근로자 4.75%
_NP_MAX_BASE = 6_370_000  # 기준소득월액 상한 (2025.7~2026.6)
_HI_RATE = 0.03595       # 건강보험 근로자 3.595%
_LTC_RATIO = 0.1314      # 장기요양보험 = 건강보험료 × 13.14%
_EI_RATE = 0.009         # 고용보험 근로자 0.9%


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _earned_income_deduction(annual_salary: int) -> int:
    """근로소득공제액 (연간)."""
    prev_upper = 0
    for upper, rate, base_deduction in _EARNED_INCOME_DEDUCTION:
        if annual_salary <= upper:
            return base_deduction + int((annual_salary - prev_upper) * rate)
        prev_upper = upper
    return 0  # unreachable


def _calculate_tax(taxable_base: int) -> int:
    """산출세액 (연간)."""
    if taxable_base <= 0:
        return 0
    for upper, rate, progressive in _TAX_BRACKETS:
        if taxable_base <= upper:
            return max(0, int(taxable_base * rate) - progressive)
    return 0  # unreachable


def _earned_income_tax_credit(calculated_tax: int, annual_salary: int) -> int:
    """근로소득세액공제 (연간, 소득세법 제59조)."""
    if calculated_tax <= 0:
        return 0

    # 공제액 계산
    if calculated_tax <= 1_300_000:
        credit = int(calculated_tax * 0.55)
    else:
        credit = 715_000 + int((calculated_tax - 1_300_000) * 0.30)

    # 한도 적용
    if annual_salary <= 33_000_000:
        limit = 740_000
    elif annual_salary <= 70_000_000:
        limit = max(660_000, 740_000 - int((annual_salary - 33_000_000) * 0.008))
    else:
        limit = max(500_000, 660_000 - int((annual_salary - 70_000_000) * 0.5))

    return min(credit, limit)


def _child_tax_credit(children: int) -> int:
    """자녀세액공제 (월 기준)."""
    if children <= 0:
        return 0
    if children <= 2:
        return _CHILD_CREDIT.get(children, 0)
    return _CHILD_CREDIT[2] + (children - 2) * _CHILD_CREDIT_EXTRA_PER


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

class IncomeTaxCalculator:
    """근로소득세 계산기 — 국세청 간이세액표 산출 공식 기반.

    Parameters
    ----------
    monthly_salary : int
        월 급여 총액 (원).
    non_taxable : int
        월 비과세 소득 (원, 기본값 0).
    dependents : int
        공제대상 가족 수, 본인 포함 (1~11, 기본값 1).
    children_8_to_20 : int
        8세~20세 자녀 수 (기본값 0).
    withholding_rate : int
        원천징수 비율 (80, 100, 120 중 택1, 기본값 100).
    """

    def __init__(
        self,
        monthly_salary: int,
        non_taxable: int = 0,
        dependents: int = 1,
        children_8_to_20: int = 0,
        withholding_rate: int = 100,
        insurance_rates: dict | None = None,
    ):
        self.monthly_salary = int(monthly_salary)
        self.non_taxable = int(non_taxable)
        self.dependents = max(1, min(int(dependents), 11))
        self.children = max(0, int(children_8_to_20))
        self.withholding_rate = max(80, min(int(withholding_rate), 120))

        # Insurance rates for tax deduction calculation
        if insurance_rates is not None:
            self._np_rate = insurance_rates.get('np_rate', _NP_RATE)
            self._np_max_base = insurance_rates.get('np_max_base', _NP_MAX_BASE)
            self._hi_rate = insurance_rates.get('hi_rate', _HI_RATE)
            self._ltc_ratio = insurance_rates.get('ltc_ratio', _LTC_RATIO)
            self._ei_rate = insurance_rates.get('ei_rate', _EI_RATE)
        else:
            self._np_rate = _NP_RATE
            self._np_max_base = _NP_MAX_BASE
            self._hi_rate = _HI_RATE
            self._ltc_ratio = _LTC_RATIO
            self._ei_rate = _EI_RATE

    def calculate(self) -> dict:
        """근로소득세 상세 산출 결과를 반환."""
        # 1) 과세기준액 (월)
        taxable_monthly = max(0, self.monthly_salary - self.non_taxable)

        # 비과세 이하이면 세액 0
        if taxable_monthly < 770_000:
            return self._zero_result(taxable_monthly)

        # 2) 연간 환산
        annual = taxable_monthly * 12

        # 3) 근로소득공제
        earned_deduction = _earned_income_deduction(annual)
        earned_income = annual - earned_deduction

        # 4) 인적공제
        personal_deduction = 1_500_000 * self.dependents

        # 5) 국민연금 공제
        np_monthly = min(taxable_monthly, self._np_max_base)
        np_annual = int(np_monthly * self._np_rate) * 12

        # 6) 건강보험 + 장기요양보험 + 고용보험 공제
        hi_monthly = int(taxable_monthly * self._hi_rate)
        ltc_monthly = int(hi_monthly * self._ltc_ratio)
        ei_monthly = int(taxable_monthly * self._ei_rate)
        insurance_annual = (hi_monthly + ltc_monthly + ei_monthly) * 12

        # 7) 과세표준
        total_deduction = personal_deduction + np_annual + insurance_annual
        taxable_base = max(0, earned_income - total_deduction)

        # 8) 산출세액
        calculated_tax = _calculate_tax(taxable_base)

        # 9) 근로소득세액공제
        earned_credit = _earned_income_tax_credit(calculated_tax, annual)

        # 10) 표준세액공제 (13만원)
        standard_credit = 130_000

        # 11) 연간 결정세액
        annual_determined = max(0, calculated_tax - earned_credit - standard_credit)

        # 12) 월 기본 세액
        monthly_base_tax = annual_determined // 12

        # 13) 자녀세액공제 (월)
        child_credit = _child_tax_credit(self.children)

        # 14) 월 근로소득세 (자녀공제 차감 후)
        monthly_income_tax = max(0, monthly_base_tax - child_credit)

        # 15) 원천징수비율 적용
        if self.withholding_rate != 100:
            monthly_income_tax = int(monthly_income_tax * self.withholding_rate / 100)

        # 16) 10원 미만 절사
        monthly_income_tax = (monthly_income_tax // 10) * 10

        # 17) 지방소득세 (소득세의 10%)
        local_tax = (int(monthly_income_tax * 0.1) // 10) * 10

        return {
            '입력정보': {
                '월급여': self.monthly_salary,
                '비과세소득': self.non_taxable,
                '과세기준액_월': taxable_monthly,
                '부양가족수': self.dependents,
                '자녀수_8세_20세': self.children,
                '원천징수비율': self.withholding_rate,
            },
            '소득공제_내역': {
                '연간_총급여': annual,
                '근로소득공제': earned_deduction,
                '근로소득금액': earned_income,
                '인적공제': personal_deduction,
                '국민연금_공제': np_annual,
                '건강보험_공제': hi_monthly * 12,
                '장기요양보험_공제': ltc_monthly * 12,
                '고용보험_공제': ei_monthly * 12,
                '소득공제_합계': total_deduction,
                '과세표준': taxable_base,
            },
            '세액_산출': {
                '산출세액': calculated_tax,
                '근로소득세액공제': earned_credit,
                '표준세액공제': standard_credit,
                '연간_결정세액': annual_determined,
                '월_기본세액': monthly_base_tax,
                '자녀세액공제_월': child_credit,
            },
            '최종_세액': {
                '근로소득세': monthly_income_tax,
                '지방소득세': local_tax,
                '합계': monthly_income_tax + local_tax,
            },
        }

    def _zero_result(self, taxable_monthly: int) -> dict:
        return {
            '입력정보': {
                '월급여': self.monthly_salary,
                '비과세소득': self.non_taxable,
                '과세기준액_월': taxable_monthly,
                '부양가족수': self.dependents,
                '자녀수_8세_20세': self.children,
                '원천징수비율': self.withholding_rate,
            },
            '소득공제_내역': {
                '연간_총급여': taxable_monthly * 12,
                '근로소득공제': 0,
                '근로소득금액': 0,
                '인적공제': 0,
                '국민연금_공제': 0,
                '건강보험_공제': 0,
                '장기요양보험_공제': 0,
                '고용보험_공제': 0,
                '소득공제_합계': 0,
                '과세표준': 0,
            },
            '세액_산출': {
                '산출세액': 0,
                '근로소득세액공제': 0,
                '표준세액공제': 0,
                '연간_결정세액': 0,
                '월_기본세액': 0,
                '자녀세액공제_월': 0,
            },
            '최종_세액': {
                '근로소득세': 0,
                '지방소득세': 0,
                '합계': 0,
            },
        }


# -----------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------
if __name__ == '__main__':
    def _fmt(n):
        return f"{n:,}"

    test_cases = [
        {'monthly_salary': 3_000_000, 'dependents': 1, 'desc': '월급 300만, 부양 1'},
        {'monthly_salary': 3_200_000, 'dependents': 4, 'desc': '월급 320만, 부양 4'},
        {'monthly_salary': 5_000_000, 'dependents': 1, 'desc': '월급 500만, 부양 1'},
        {'monthly_salary': 5_000_000, 'dependents': 3, 'children_8_to_20': 1,
         'desc': '월급 500만, 부양 3, 자녀 1'},
        {'monthly_salary': 10_000_000, 'dependents': 1, 'desc': '월급 1000만, 부양 1'},
        {'monthly_salary': 3_000_000, 'dependents': 1, 'withholding_rate': 80,
         'desc': '월급 300만, 80% 원천징수'},
    ]

    for tc in test_cases:
        desc = tc.get('desc', '')
        tc_args = {k: v for k, v in tc.items() if k != 'desc'}
        calc = IncomeTaxCalculator(**tc_args)
        r = calc.calculate()
        tax = r['최종_세액']
        detail = r['소득공제_내역']
        print(f"[{desc}]")
        print(f"  과세표준: {_fmt(detail['과세표준'])}원")
        print(f"  산출세액: {_fmt(r['세액_산출']['산출세액'])}원/년")
        print(f"  근로소득세: {_fmt(tax['근로소득세'])}원/월")
        print(f"  지방소득세: {_fmt(tax['지방소득세'])}원/월")
        print(f"  합계: {_fmt(tax['합계'])}원/월")
        print()
