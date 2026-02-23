"""Calculator wrapper functions and OpenAI tool definitions for function calling."""

import json
import logging

from calculator import (
    WageCalculator, InsuranceCalculator, CompanySize, IndustryType,
    RetirementPayCalculator, AnnualLeaveCalculator, IncomeTaxCalculator,
)
from calculator.rates import (
    get_wage_insurance_rates,
    get_insurance_rates,
    get_income_tax_rates,
)


def calculate_wage(
    salary_type: str,
    amount: int,
    tax_free_monthly: int = 0,
    dependents: int = 1,
    children_8_to_20: int = 0,
    company_size: str = 'small'
) -> dict:
    """임금 계산 (실수령액, 4대보험료, 세금)"""
    calc = WageCalculator(rates=get_wage_insurance_rates())

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
    calc = InsuranceCalculator(rates=get_insurance_rates())

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


def calculate_retirement_pay(
    start_date: str,
    end_date: str,
    monthly_basic_pay,
    monthly_other_pay=0,
    annual_bonus: int = 0,
    annual_leave_pay: int = 0,
    excluded_days_avg: int = 0,
    excluded_days_service: int = 0,
    ordinary_daily_wage: int | None = None,
) -> dict:
    """퇴직금 계산 (고용노동부 공식 기준)"""
    try:
        calc = RetirementPayCalculator(
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
        return calc.calculate()
    except ValueError as e:
        return {"error": str(e)}


def calculate_annual_leave(
    hire_date: str,
    end_date: str | None = None,
) -> dict:
    """연차유급휴가 계산 (근로기준법 제60조 기준)"""
    try:
        calc = AnnualLeaveCalculator(
            hire_date=hire_date,
            end_date=end_date,
        )
        return calc.calculate()
    except ValueError as e:
        return {"error": str(e)}


def calculate_income_tax(
    monthly_salary: int,
    non_taxable: int = 0,
    dependents: int = 1,
    children_8_to_20: int = 0,
    withholding_rate: int = 100,
) -> dict:
    """근로소득세 계산 (국세청 간이세액표 산출 공식 기준)"""
    calc = IncomeTaxCalculator(
        monthly_salary=monthly_salary,
        non_taxable=non_taxable,
        dependents=dependents,
        children_8_to_20=children_8_to_20,
        withholding_rate=withholding_rate,
        insurance_rates=get_income_tax_rates(),
    )
    return calc.calculate()


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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_retirement_pay",
            "description": "퇴직금 계산 (고용노동부 공식 기준). 입사일/퇴직일, 퇴직 전 3개월 임금, 상여금, 연차수당을 바탕으로 1일 평균임금과 퇴직금을 산출합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "입사일자 (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "퇴직일자 (YYYY-MM-DD, 마지막 근무일의 다음 날)"
                    },
                    "monthly_basic_pay": {
                        "oneOf": [
                            {"type": "integer", "description": "3개월 동일 기본급 (원)"},
                            {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3, "description": "퇴직 전 3개월 기본급 각각 [월1, 월2, 월3] (원)"}
                        ],
                        "description": "퇴직 전 3개월 기본급. 정수면 3개월 동일 적용, 배열이면 각 월별 입력"
                    },
                    "monthly_other_pay": {
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
                        ],
                        "description": "퇴직 전 3개월 기타수당 (기본값 0)",
                        "default": 0
                    },
                    "annual_bonus": {
                        "type": "integer",
                        "description": "연간 상여금 총액 (원)",
                        "default": 0
                    },
                    "annual_leave_pay": {
                        "type": "integer",
                        "description": "연차수당 (원)",
                        "default": 0
                    },
                    "excluded_days_avg": {
                        "type": "integer",
                        "description": "미산입기간 일수 (평균임금 산정기간 제외, 예: 육아휴직)",
                        "default": 0
                    },
                    "excluded_days_service": {
                        "type": "integer",
                        "description": "근무제외기간 일수 (근속기간 제외, 예: 개인휴직)",
                        "default": 0
                    },
                    "ordinary_daily_wage": {
                        "type": "integer",
                        "description": "1일 통상임금 (원). 평균임금보다 크면 통상임금 기준 적용"
                    }
                },
                "required": ["start_date", "end_date", "monthly_basic_pay"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_annual_leave",
            "description": "연차유급휴가 계산 (근로기준법 제60조 기준). 입사일과 기준일(또는 퇴직일)로 연도별 연차 발생일수를 계산합니다. 1년 미만 월차, 가산휴가(3년 이상 근속 시 매 2년 +1일, 최대 25일)를 포함합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hire_date": {
                        "type": "string",
                        "description": "입사일자 (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "기준일자 또는 퇴직일자 (YYYY-MM-DD). 미입력 시 오늘 날짜 적용"
                    }
                },
                "required": ["hire_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_income_tax",
            "description": "근로소득세 계산 (국세청 간이세액표 산출 공식 기준). 월급여, 부양가족 수, 자녀 수를 입력하면 소득세와 지방소득세를 계산합니다. 소득공제, 과세표준, 산출세액, 세액공제 전 과정을 상세히 보여줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_salary": {
                        "type": "integer",
                        "description": "월 급여 총액 (원 단위)"
                    },
                    "non_taxable": {
                        "type": "integer",
                        "description": "월 비과세 소득 (식대 등, 원 단위)",
                        "default": 0
                    },
                    "dependents": {
                        "type": "integer",
                        "description": "공제대상 가족 수 (본인 포함, 1~11)",
                        "default": 1
                    },
                    "children_8_to_20": {
                        "type": "integer",
                        "description": "8세~20세 자녀 수 (자녀세액공제 대상)",
                        "default": 0
                    },
                    "withholding_rate": {
                        "type": "integer",
                        "enum": [80, 100, 120],
                        "description": "원천징수 비율 (80%, 100%, 120% 중 택1)",
                        "default": 100
                    }
                },
                "required": ["monthly_salary"]
            }
        }
    }
]


_SENSITIVE_KEYS = frozenset({
    'amount', 'monthly_income', 'non_taxable', 'tax_free_monthly',
    'monthly_salary', 'monthly_basic_pay', 'monthly_other_pay',
    'annual_bonus', 'annual_leave_pay', 'ordinary_daily_wage',
})


def handle_tool_calls(messages, response_message):
    """Execute tool calls and return (calculation_results, updated_messages).

    Creates a shallow copy of *messages* so the caller's original list is not
    modified.  The returned ``updated_messages`` contains the appended entries.
    """
    messages = list(messages)
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

        safe_args = {k: '***' if k in _SENSITIVE_KEYS else v
                     for k, v in function_args.items()}
        logging.info("[Function Call] %s with args: %s", function_name, safe_args)
        try:
            if function_name == "calculate_wage":
                function_response = calculate_wage(**function_args)
            elif function_name == "calculate_insurance":
                function_response = calculate_insurance(**function_args)
            elif function_name == "calculate_retirement_pay":
                function_response = calculate_retirement_pay(**function_args)
            elif function_name == "calculate_annual_leave":
                function_response = calculate_annual_leave(**function_args)
            elif function_name == "calculate_income_tax":
                function_response = calculate_income_tax(**function_args)
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
