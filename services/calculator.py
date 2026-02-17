"""Calculator wrapper functions and OpenAI tool definitions for function calling."""

import json
import logging

from calculator import WageCalculator, InsuranceCalculator, CompanySize, IndustryType


def calculate_wage(
    salary_type: str,
    amount: int,
    tax_free_monthly: int = 0,
    dependents: int = 1,
    children_8_to_20: int = 0,
    company_size: str = 'small'
) -> dict:
    """임금 계산 (실수령액, 4대보험료, 세금)"""
    calc = WageCalculator()

    if salary_type == '연봉':
        result = calc.calculate_from_annual(
            annual_salary=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size
        )
    elif salary_type == '월급':
        result = calc.calculate_from_monthly(
            monthly_salary=amount,
            tax_free_monthly=tax_free_monthly,
            dependents=dependents,
            children_8_to_20=children_8_to_20,
            company_size=company_size
        )
    else:
        return {"error": f"잘못된 급여 유형: {salary_type}"}

    return result


def calculate_insurance(
    monthly_income: int,
    non_taxable: int = 0,
    company_size_code: str = 'UNDER_150',
    industry_code: str = 'OTHERS'
) -> dict:
    """4대보험료 계산 (국민연금, 건강보험, 장기요양보험, 고용보험, 산재보험)"""
    calc = InsuranceCalculator()

    size_map = {
        'UNDER_150': CompanySize.UNDER_150,
        'PRIORITY_SUPPORT': CompanySize.PRIORITY_SUPPORT,
        'FROM_150_TO_999': CompanySize.FROM_150_TO_999,
        'OVER_1000': CompanySize.OVER_1000
    }
    company_size = size_map.get(company_size_code, CompanySize.UNDER_150)

    industry = getattr(IndustryType, industry_code, None)
    if industry is None:
        logging.warning("Unknown industry_code: %s, falling back to OTHERS", industry_code)
        industry = IndustryType.OTHERS

    result = calc.calculate_all(
        monthly_income=monthly_income,
        non_taxable=non_taxable,
        company_size=company_size,
        industry=industry
    )

    return result


# GPT Function definitions for Function Calling
CALCULATOR_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_wage",
            "description": "한국 노동법에 따른 임금 계산 (실수령액, 4대보험료, 소득세, 지방소득세). 연봉이나 월급을 입력하면 실제 받게 되는 급여를 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "salary_type": {
                        "type": "string",
                        "enum": ["연봉", "월급"],
                        "description": "급여 유형"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "급여액 (원 단위)"
                    },
                    "tax_free_monthly": {
                        "type": "integer",
                        "description": "월 비과세액 (식대 등, 최대 200,000원)",
                        "default": 0
                    },
                    "dependents": {
                        "type": "integer",
                        "description": "부양가족 수 (본인 포함)",
                        "default": 1
                    },
                    "children_8_to_20": {
                        "type": "integer",
                        "description": "8~20세 자녀 수 (자녀세액공제 대상)",
                        "default": 0
                    },
                    "company_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "회사 규모: small(150인 미만), medium(150~999인), large(1000인 이상)",
                        "default": "small"
                    }
                },
                "required": ["salary_type", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_insurance",
            "description": "한국 4대보험료 상세 계산 (국민연금, 건강보험, 장기요양보험, 고용보험, 산재보험). 근로자와 사업주가 각각 부담하는 보험료를 업종과 회사 규모에 따라 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_income": {
                        "type": "integer",
                        "description": "월 소득 (원 단위)"
                    },
                    "non_taxable": {
                        "type": "integer",
                        "description": "비과세소득 (원 단위)",
                        "default": 0
                    },
                    "company_size_code": {
                        "type": "string",
                        "enum": ["UNDER_150", "PRIORITY_SUPPORT", "FROM_150_TO_999", "OVER_1000"],
                        "description": "회사 규모: UNDER_150(150인 미만), PRIORITY_SUPPORT(우선지원대상), FROM_150_TO_999(150~999인), OVER_1000(1000인 이상)",
                        "default": "UNDER_150"
                    },
                    "industry_code": {
                        "type": "string",
                        "description": "산재보험 업종 코드 (예: OTHERS-기타사업, WHOLESALE_RETAIL-도소매, PROFESSIONAL-전문서비스, FINANCE_INSURANCE-금융, CONSTRUCTION-건설)",
                        "default": "OTHERS"
                    }
                },
                "required": ["monthly_income"]
            }
        }
    }
]


def handle_tool_calls(messages, response_message):
    """Execute tool calls and return (calculation_results, updated_messages).

    Note: Mutates ``messages`` in place by appending tool call/response entries.
    """
    calculation_results = []
    messages.append(response_message)

    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logging.error("[Function Call] Invalid JSON args for %s: %s", function_name, e)
            function_response = {"error": f"잘못된 함수 인자 형식: {str(e)}"}
            calculation_results.append({'function': function_name, 'args': {}, 'result': function_response})
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False, indent=2)
            })
            continue

        safe_args = {k: '***' if k in ('amount', 'monthly_income', 'non_taxable', 'tax_free_monthly') else v
                     for k, v in function_args.items()}
        logging.info(f"[Function Call] {function_name} with args: {safe_args}")
        try:
            if function_name == "calculate_wage":
                function_response = calculate_wage(**function_args)
            elif function_name == "calculate_insurance":
                function_response = calculate_insurance(**function_args)
            else:
                function_response = {"error": "Unknown function"}
        except Exception as e:
            logging.error("[Function Call] Execution error in %s: %s", function_name, e)
            function_response = {"error": f"함수 실행 오류: {str(e)}"}

        calculation_results.append({'function': function_name, 'args': safe_args, 'result': function_response})
        messages.append({
            "role": "tool", "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_response, ensure_ascii=False, indent=2)
        })

    return calculation_results, messages
