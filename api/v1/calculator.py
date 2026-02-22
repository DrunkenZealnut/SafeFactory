"""Standalone calculator endpoints for wage and insurance calculations."""

import logging

from flask import request

from api.v1 import v1_bp
from api.response import success_response, error_response
from services.calculator import (
    calculate_wage, calculate_insurance, calculate_retirement_pay,
    calculate_annual_leave, calculate_income_tax,
)

VALID_COMPANY_SIZES = {'small', 'medium', 'large'}
VALID_COMPANY_SIZE_CODES = {'UNDER_150', 'PRIORITY_SUPPORT', 'FROM_150_TO_999', 'OVER_1000'}
VALID_INDUSTRY_CODES = {
    'MINING_COAL', 'MINING_METAL', 'MINING_OTHER', 'FOOD_BEVERAGE', 'TEXTILE',
    'WOOD_PAPER', 'CHEMICAL', 'CEMENT', 'METAL_BASIC', 'METAL_FABRICATION',
    'MACHINERY', 'ELECTRONICS', 'ELECTRICAL', 'PRECISION', 'TRANSPORT_EQUIP',
    'SHIP_BUILDING', 'OTHER_MANUFACTURING', 'ELECTRICITY_GAS_WATER',
    'CONSTRUCTION', 'TRANSPORT_STORAGE', 'TELECOM', 'FINANCE_INSURANCE',
    'PROFESSIONAL', 'WHOLESALE_RETAIL', 'REAL_ESTATE', 'EDUCATION',
    'HEALTH_SOCIAL', 'ENTERTAINMENT', 'OTHERS',
}


@v1_bp.route('/calculate/wage', methods=['POST'])
def api_calculate_wage():
    """Calculate net salary, taxes, and social insurance from gross salary."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        salary_type = data.get('salary_type', '')
        if salary_type not in ('연봉', '월급'):
            return error_response('salary_type은 "연봉" 또는 "월급"이어야 합니다.', 400)

        try:
            amount = int(data.get('amount', 0))
        except (ValueError, TypeError):
            return error_response('amount는 정수여야 합니다.', 400)
        if amount <= 0:
            return error_response('amount는 0보다 커야 합니다.', 400)

        try:
            tax_free_monthly = int(data.get('tax_free_monthly', 0))
            dependents = int(data.get('dependents', 1))
            children_8_to_20 = int(data.get('children_8_to_20', 0))
        except (ValueError, TypeError):
            return error_response('잘못된 숫자 형식입니다.', 400)
        if tax_free_monthly < 0 or dependents < 0 or children_8_to_20 < 0:
            return error_response('tax_free_monthly, dependents, children_8_to_20은 음수일 수 없습니다.', 400)

        company_size = data.get('company_size', 'small')
        if company_size not in VALID_COMPANY_SIZES:
            return error_response(
                f'company_size는 {", ".join(sorted(VALID_COMPANY_SIZES))} 중 하나여야 합니다.', 400)

        result = calculate_wage(
            salary_type=salary_type,
            amount=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size,
        )

        return success_response(data=result)

    except Exception:
        logging.exception('Calculation failed')
        return error_response('계산 중 오류가 발생했습니다.', 500)


@v1_bp.route('/calculate/insurance', methods=['POST'])
def api_calculate_insurance():
    """Calculate detailed social insurance premiums."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        try:
            monthly_income = int(data.get('monthly_income', 0))
        except (ValueError, TypeError):
            return error_response('monthly_income은 정수여야 합니다.', 400)
        if monthly_income <= 0:
            return error_response('monthly_income은 0보다 커야 합니다.', 400)

        try:
            non_taxable = int(data.get('non_taxable', 0))
        except (ValueError, TypeError):
            return error_response('잘못된 숫자 형식입니다.', 400)
        if non_taxable < 0:
            return error_response('non_taxable은 음수일 수 없습니다.', 400)

        company_size_code = data.get('company_size_code', 'UNDER_150')
        if company_size_code not in VALID_COMPANY_SIZE_CODES:
            return error_response(
                f'company_size_code는 {", ".join(sorted(VALID_COMPANY_SIZE_CODES))} 중 하나여야 합니다.', 400)

        industry_code = data.get('industry_code', 'OTHERS')
        if industry_code not in VALID_INDUSTRY_CODES:
            return error_response(
                f'유효하지 않은 industry_code입니다: {industry_code}', 400)

        result = calculate_insurance(
            monthly_income=monthly_income,
            non_taxable=non_taxable,
            company_size_code=company_size_code,
            industry_code=industry_code,
        )

        return success_response(data=result)

    except Exception:
        logging.exception('Calculation failed')
        return error_response('계산 중 오류가 발생했습니다.', 500)


@v1_bp.route('/calculate/retirement', methods=['POST'])
def api_calculate_retirement():
    """Calculate retirement pay (퇴직금) based on Korean labor law."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        start_date = data.get('start_date')
        end_date = data.get('end_date')
        if not start_date or not end_date:
            return error_response('start_date와 end_date는 필수입니다. (YYYY-MM-DD)', 400)

        monthly_basic_pay = data.get('monthly_basic_pay')
        if not monthly_basic_pay:
            return error_response('monthly_basic_pay는 필수입니다.', 400)
        try:
            monthly_basic_pay = int(monthly_basic_pay)
        except (ValueError, TypeError):
            return error_response('monthly_basic_pay는 정수여야 합니다.', 400)

        try:
            monthly_other_pay = int(data.get('monthly_other_pay', 0))
        except (ValueError, TypeError):
            return error_response('monthly_other_pay는 정수여야 합니다.', 400)

        try:
            annual_bonus = int(data.get('annual_bonus', 0))
            annual_leave_pay = int(data.get('annual_leave_pay', 0))
            excluded_days_avg = int(data.get('excluded_days_avg', 0))
            excluded_days_service = int(data.get('excluded_days_service', 0))
        except (ValueError, TypeError):
            return error_response('잘못된 숫자 형식입니다.', 400)

        ordinary_daily_wage = data.get('ordinary_daily_wage')
        if ordinary_daily_wage is not None:
            try:
                ordinary_daily_wage = int(ordinary_daily_wage)
            except (ValueError, TypeError):
                return error_response('ordinary_daily_wage는 정수여야 합니다.', 400)

        result = calculate_retirement_pay(
            start_date=start_date,
            end_date=end_date,
            monthly_basic_pay=monthly_basic_pay,
            monthly_other_pay=monthly_other_pay,
            annual_bonus=annual_bonus,
            annual_leave_pay=annual_leave_pay,
            excluded_days_avg=excluded_days_avg,
            excluded_days_service=excluded_days_service,
            ordinary_daily_wage=ordinary_daily_wage,
        )

        if 'error' in result:
            return error_response(result['error'], 400)

        return success_response(data=result)

    except Exception:
        logging.exception('Retirement pay calculation failed')
        return error_response('퇴직금 계산 중 오류가 발생했습니다.', 500)


@v1_bp.route('/calculate/annual-leave', methods=['POST'])
def api_calculate_annual_leave():
    """Calculate annual leave entitlement (연차유급휴가) based on Korean labor law."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        hire_date = data.get('hire_date')
        if not hire_date:
            return error_response('hire_date는 필수입니다. (YYYY-MM-DD)', 400)

        end_date = data.get('end_date')  # optional

        result = calculate_annual_leave(
            hire_date=hire_date,
            end_date=end_date,
        )

        if 'error' in result:
            return error_response(result['error'], 400)

        return success_response(data=result)

    except Exception:
        logging.exception('Annual leave calculation failed')
        return error_response('연차휴가 계산 중 오류가 발생했습니다.', 500)


@v1_bp.route('/calculate/income-tax', methods=['POST'])
def api_calculate_income_tax():
    """Calculate income tax (근로소득세) based on NTS simplified tax table formula."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        try:
            monthly_salary = int(data.get('monthly_salary', 0))
        except (ValueError, TypeError):
            return error_response('monthly_salary는 정수여야 합니다.', 400)
        if monthly_salary <= 0:
            return error_response('monthly_salary는 0보다 커야 합니다.', 400)

        try:
            non_taxable = int(data.get('non_taxable', 0))
            dependents = int(data.get('dependents', 1))
            children_8_to_20 = int(data.get('children_8_to_20', 0))
            withholding_rate = int(data.get('withholding_rate', 100))
        except (ValueError, TypeError):
            return error_response('잘못된 숫자 형식입니다.', 400)

        if non_taxable < 0:
            return error_response('non_taxable은 음수일 수 없습니다.', 400)
        if dependents < 0 or children_8_to_20 < 0:
            return error_response('dependents, children_8_to_20은 음수일 수 없습니다.', 400)
        if withholding_rate not in (80, 100, 120):
            return error_response('withholding_rate는 80, 100, 120 중 하나여야 합니다.', 400)

        result = calculate_income_tax(
            monthly_salary=monthly_salary,
            non_taxable=non_taxable,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            withholding_rate=withholding_rate,
        )

        return success_response(data=result)

    except Exception:
        logging.exception('Income tax calculation failed')
        return error_response('근로소득세 계산 중 오류가 발생했습니다.', 500)
