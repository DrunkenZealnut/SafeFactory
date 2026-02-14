"""Standalone calculator endpoints for wage and insurance calculations."""

from flask import request

from api.v1 import v1_bp
from api.response import success_response, error_response
from services.calculator import calculate_wage, calculate_insurance


@v1_bp.route('/calculate/wage', methods=['POST'])
def api_calculate_wage():
    """Calculate net salary, taxes, and social insurance from gross salary."""
    try:
        data = request.get_json()
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

        result = calculate_wage(
            salary_type=salary_type,
            amount=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size,
        )

        return success_response(data=result)

    except Exception as e:
        return error_response(str(e), 500)


@v1_bp.route('/calculate/insurance', methods=['POST'])
def api_calculate_insurance():
    """Calculate detailed social insurance premiums."""
    try:
        data = request.get_json()
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
        industry_code = data.get('industry_code', 'OTHERS')

        result = calculate_insurance(
            monthly_income=monthly_income,
            non_taxable=non_taxable,
            company_size_code=company_size_code,
            industry_code=industry_code,
        )

        return success_response(data=result)

    except Exception as e:
        return error_response(str(e), 500)
