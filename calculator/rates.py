"""Centralized rate loading with DB-override support.

Loads calculator rates from SystemSetting DB (via services/settings),
falling back to hardcoded defaults if no DB row exists.

Usage:
    from calculator.rates import get_insurance_rates, get_minimum_wage
    rates = get_insurance_rates()
    calc = InsuranceCalculator(rates=rates)
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded defaults (2026 기준) — DB에 없으면 이 값 사용
# ⚠️ 매년 1월 갱신 필요: 4대보험료율, 최저임금, 기준소득월액 상·하한
#   - 갱신 방법 1: 관리자 페이지에서 SystemSettings DB 값 수정 (재시작 불필요)
#   - 갱신 방법 2: 아래 _DEFAULTS dict 직접 수정 후 배포
#   - 출처: 4대사회보험정보연계센터, 고용노동부 최저임금 고시
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, str] = {
    # 국민연금
    'calc.np_rate': '0.0475',
    'calc.np_max_income': '6370000',
    'calc.np_min_income': '400000',
    # 건강보험
    'calc.hi_rate': '0.03595',
    'calc.hi_max_income': '127725730',
    'calc.hi_min_income': '280528',
    'calc.hi_max_premium': '9183460',
    'calc.hi_min_premium': '20160',
    # 장기요양보험
    'calc.ltc_rate': '0.1314',
    # 고용보험
    'calc.ei_employee': '0.009',
    'calc.ei_employer_base': '0.009',
    'calc.ei_under_150': '0.0025',
    'calc.ei_priority': '0.0045',
    'calc.ei_150_to_999': '0.0065',
    'calc.ei_over_1000': '0.0085',
    # 산재보험 부가금
    'calc.ia_commute': '0.006',
    'calc.ia_wage_claim': '0.0006',
    'calc.ia_asbestos': '0.0003',
    # 최저임금
    'calc.min_wage_year': '2026',
    'calc.min_wage_2026': '10320',
    'calc.min_wage_2025': '10030',
    'calc.min_wage_2024': '9860',
    'calc.min_wage_2023': '9620',
    'calc.min_wage_2022': '9160',
    'calc.min_wage_2021': '8720',
    'calc.min_wage_2020': '8590',
    # 메타
    'calc.rates_updated_at': '2026-01-01',
    'calc.rates_year': '2026',
}


def _get(key: str) -> str:
    """Get a rate value: DB setting → module default."""
    try:
        from services.settings import get_setting
        val = get_setting(key)
        if val is not None:
            return val
    except Exception:
        logger.debug("Failed to fetch setting %s from DB, using default", key)
    return _DEFAULTS.get(key, '')


def _float(key: str) -> float:
    try:
        return float(_get(key))
    except (ValueError, TypeError):
        return float(_DEFAULTS.get(key, '0'))


def _int(key: str) -> int:
    try:
        return int(_get(key))
    except (ValueError, TypeError):
        return int(_DEFAULTS.get(key, '0'))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_insurance_rates():
    """Load InsuranceRates2026 for InsuranceCalculator."""
    from calculator.insurance_calculator import InsuranceRates2026

    rate = _float('calc.np_rate')
    hi_rate = _float('calc.hi_rate')

    return InsuranceRates2026(
        national_pension_total=rate * 2,
        national_pension_employee=rate,
        national_pension_employer=rate,
        national_pension_max_income=_int('calc.np_max_income'),
        national_pension_min_income=_int('calc.np_min_income'),
        health_insurance_total=hi_rate * 2,
        health_insurance_employee=hi_rate,
        health_insurance_employer=hi_rate,
        health_insurance_max_income=_int('calc.hi_max_income'),
        health_insurance_min_income=_int('calc.hi_min_income'),
        health_insurance_max_premium=_int('calc.hi_max_premium'),
        health_insurance_min_premium=_int('calc.hi_min_premium'),
        long_term_care_rate=_float('calc.ltc_rate'),
        employment_insurance_employee=_float('calc.ei_employee'),
        employment_insurance_employer_base=_float('calc.ei_employer_base'),
        employment_stability_under_150=_float('calc.ei_under_150'),
        employment_stability_priority=_float('calc.ei_priority'),
        employment_stability_150_to_999=_float('calc.ei_150_to_999'),
        employment_stability_over_1000=_float('calc.ei_over_1000'),
        commute_accident_rate=_float('calc.ia_commute'),
        wage_claim_rate=_float('calc.ia_wage_claim'),
        asbestos_rate=_float('calc.ia_asbestos'),
    )


def get_wage_insurance_rates():
    """Load InsuranceRates for WageCalculator."""
    from calculator.wage_calculator import InsuranceRates

    rate = _float('calc.np_rate')
    hi_rate = _float('calc.hi_rate')

    return InsuranceRates(
        national_pension_employee=rate,
        national_pension_employer=rate,
        health_insurance_employee=hi_rate,
        health_insurance_employer=hi_rate,
        long_term_care_rate=_float('calc.ltc_rate'),
        employment_insurance_employee=_float('calc.ei_employee'),
        employment_insurance_employer_small=_float('calc.ei_employer_base')
            + _float('calc.ei_under_150'),
        employment_insurance_employer_medium=_float('calc.ei_employer_base')
            + _float('calc.ei_150_to_999'),
        employment_insurance_employer_large=_float('calc.ei_employer_base')
            + _float('calc.ei_over_1000'),
    )


def get_minimum_wage(year: int | None = None) -> int:
    """Get minimum hourly wage for the given year."""
    if year is None:
        year = get_current_wage_year()
    val = _get(f'calc.min_wage_{year}')
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    # Fallback to module table
    from calculator.minimum_wage import MINIMUM_WAGE_TABLE, CURRENT_MIN_HOURLY
    return MINIMUM_WAGE_TABLE.get(year, CURRENT_MIN_HOURLY)


def get_current_wage_year() -> int:
    """Get the effective minimum wage year."""
    try:
        return int(_get('calc.min_wage_year'))
    except (ValueError, TypeError):
        return 2026


def get_income_tax_rates() -> dict:
    """Return insurance rates needed by IncomeTaxCalculator."""
    return {
        'np_rate': _float('calc.np_rate'),
        'np_max_base': _int('calc.np_max_income'),
        'hi_rate': _float('calc.hi_rate'),
        'ltc_ratio': _float('calc.ltc_rate'),
        'ei_rate': _float('calc.ei_employee'),
    }


def get_rates_freshness() -> dict:
    """Return metadata about rate freshness for staleness warnings."""
    updated_at = _get('calc.rates_updated_at')
    effective_year = _get('calc.rates_year')
    current_year = date.today().year

    is_stale = False
    warning = None

    if effective_year:
        try:
            if int(effective_year) < current_year:
                is_stale = True
                warning = (
                    f"적용 요율이 {effective_year}년 기준입니다. "
                    f"현재 {current_year}년 요율로 업데이트가 필요할 수 있습니다."
                )
        except ValueError:
            pass

    return {
        'rates_updated_at': updated_at or '미등록',
        'effective_year': effective_year,
        'is_stale': is_stale,
        'warning': warning,
    }


def get_all_rates() -> dict:
    """Return all current rates as a dict (for admin display)."""
    return {key: _get(key) for key in _DEFAULTS}
