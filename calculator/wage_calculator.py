"""
한국 임금계산기 (급여명세서 계산)
- 연봉/월급 기반 실수령액 계산 (세전→세후)
- 세후 역산: 희망 실수령액으로부터 필요 세전 급여 산출
- 4대보험료 및 근로소득세 자동 계산
- 2026년 기준 요율 적용 (참고: https://www.4insure.or.kr)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InsuranceRates:
    """4대보험 요율 (2026년 기준, 참고: https://www.4insure.or.kr)"""
    # 국민연금: 총 9.5%, 근로자 4.75%
    national_pension_employee: float = 0.0475
    national_pension_employer: float = 0.0475

    # 건강보험: 총 7.19%, 근로자 3.595%
    health_insurance_employee: float = 0.03595
    health_insurance_employer: float = 0.03595

    # 장기요양보험: 건강보험료의 13.14% (0.9448% ÷ 7.19%)
    long_term_care_rate: float = 0.1314

    # 고용보험: 근로자 0.9%, 사업주 0.9%~1.65% (규모별)
    employment_insurance_employee: float = 0.009
    employment_insurance_employer_small: float = 0.009   # 150인 미만
    employment_insurance_employer_medium: float = 0.011  # 150~999인
    employment_insurance_employer_large: float = 0.013   # 1000인 이상, 공기업

    # 산재보험: 전액 사업주 부담 (업종별 상이, 평균 약 1.47%)
    industrial_accident_employer: float = 0.0147


class WageCalculator:
    """임금계산기 클래스"""

    # 근로소득세 간이세액표 기반 계산을 위한 구간 (월급여 기준, 부양가족 1인)
    # 실제로는 국세청 간이세액표 전체를 사용해야 하지만 주요 구간만 구현
    INCOME_TAX_TABLE = {
        # (월급여 하한, 상한): (부양가족 1인 기준 세액)
        (0, 1060000): 0,
        (1060000, 1500000): 9000,
        (1500000, 2000000): 19000,
        (2000000, 2500000): 46000,
        (2500000, 3000000): 79000,
        (3000000, 3500000): 117000,
        (3500000, 4000000): 161000,
        (4000000, 4500000): 206000,
        (4500000, 5000000): 256000,
        (5000000, 5500000): 314000,
        (5500000, 6000000): 374000,
        (6000000, 6500000): 434000,
        (6500000, 7000000): 501000,
        (7000000, 7500000): 575000,
        (7500000, 8000000): 652000,
        (8000000, 8500000): 729000,
        (8500000, 9000000): 811000,
        (9000000, 9500000): 896000,
        (9500000, 10000000): 984000,
        (10000000, 15000000): 1500000,
        (15000000, 20000000): 2600000,
        (20000000, 30000000): 4200000,
        (30000000, float('inf')): 7000000,
    }

    # 부양가족 수에 따른 세액 공제 (1인당 약 12,500원 추가 공제)
    DEPENDENT_DEDUCTION_PER_PERSON = 12500

    # 자녀세액공제 (8세~20세 자녀 1명당)
    CHILD_TAX_CREDIT = {
        1: 12500,
        2: 29160,
        3: 54160,  # 3명 이상부터 추가 공제
    }

    def __init__(self, rates: Optional[InsuranceRates] = None):
        self.rates = rates or InsuranceRates()

    def calculate_national_pension(self, monthly_salary: int) -> tuple[int, int]:
        """
        국민연금 계산
        - 기준소득월액 상한: 6,370,000원 (2025.7~2026.6)
        - 기준소득월액 하한: 400,000원
        """
        MIN_BASE = 400000
        MAX_BASE = 6370000

        base_salary = max(MIN_BASE, min(monthly_salary, MAX_BASE))

        employee = int(base_salary * self.rates.national_pension_employee)
        employer = int(base_salary * self.rates.national_pension_employer)

        # 원 단위 절사 (10원 미만)
        employee = (employee // 10) * 10
        employer = (employer // 10) * 10

        return employee, employer

    def calculate_health_insurance(self, monthly_salary: int) -> tuple[int, int]:
        """
        건강보험 계산
        - 보수월액 기준
        """
        employee = int(monthly_salary * self.rates.health_insurance_employee)
        employer = int(monthly_salary * self.rates.health_insurance_employer)

        # 10원 미만 절사
        employee = (employee // 10) * 10
        employer = (employer // 10) * 10

        return employee, employer

    def calculate_long_term_care(self, health_insurance_employee: int,
                                  health_insurance_employer: int) -> tuple[int, int]:
        """
        장기요양보험 계산
        - 건강보험료의 일정 비율
        """
        employee = int(health_insurance_employee * self.rates.long_term_care_rate)
        employer = int(health_insurance_employer * self.rates.long_term_care_rate)

        # 10원 미만 절사
        employee = (employee // 10) * 10
        employer = (employer // 10) * 10

        return employee, employer

    def calculate_employment_insurance(self, monthly_salary: int,
                                        company_size: str = 'small') -> tuple[int, int]:
        """
        고용보험 계산
        - company_size: 'small' (150인 미만), 'medium' (150~999인), 'large' (1000인 이상)
        """
        employee = int(monthly_salary * self.rates.employment_insurance_employee)

        if company_size == 'small':
            employer_rate = self.rates.employment_insurance_employer_small
        elif company_size == 'medium':
            employer_rate = self.rates.employment_insurance_employer_medium
        else:
            employer_rate = self.rates.employment_insurance_employer_large

        employer = int(monthly_salary * employer_rate)

        # 10원 미만 절사
        employee = (employee // 10) * 10
        employer = (employer // 10) * 10

        return employee, employer

    def calculate_industrial_accident(self, monthly_salary: int) -> int:
        """
        산재보험 계산 (전액 사업주 부담)
        """
        employer = int(monthly_salary * self.rates.industrial_accident_employer)
        return (employer // 10) * 10

    def calculate_income_tax(self, monthly_salary: int,
                              tax_free_amount: int = 0,
                              dependents: int = 1,
                              children_8_to_20: int = 0) -> int:
        """
        근로소득세 계산 (간이세액표 기반)

        Args:
            monthly_salary: 월 급여액
            tax_free_amount: 비과세액 (식대 등)
            dependents: 부양가족 수 (본인 포함)
            children_8_to_20: 8세~20세 자녀 수
        """
        # 과세 대상 급여
        taxable_salary = monthly_salary - tax_free_amount

        if taxable_salary < 1060000:
            return 0

        # 기본 세액 조회
        base_tax = 0
        for (lower, upper), tax in self.INCOME_TAX_TABLE.items():
            if lower <= taxable_salary < upper:
                base_tax = tax
                break

        # 부양가족 공제 (본인 제외한 추가 부양가족에 대해)
        dependent_deduction = max(0, dependents - 1) * self.DEPENDENT_DEDUCTION_PER_PERSON

        # 자녀세액공제
        child_credit = 0
        if children_8_to_20 > 0:
            if children_8_to_20 <= 2:
                child_credit = self.CHILD_TAX_CREDIT.get(children_8_to_20, 0)
            else:
                # 3명 이상: 기본 + 추가
                child_credit = self.CHILD_TAX_CREDIT[2] + (children_8_to_20 - 2) * 25000

        # 최종 소득세
        income_tax = max(0, base_tax - dependent_deduction - child_credit)

        return (income_tax // 10) * 10

    def calculate_local_income_tax(self, income_tax: int) -> int:
        """
        지방소득세 계산
        - 소득세의 10%
        """
        local_tax = int(income_tax * 0.1)
        return (local_tax // 10) * 10  # 10원 단위로 절사

    def calculate_from_net(self, target_net_monthly: int,
                            tax_free_monthly: int = 0,
                            dependents: int = 1,
                            children_8_to_20: int = 0,
                            company_size: str = 'small') -> dict:
        """세후(실수령) 금액으로부터 필요 세전(월급) 역산 (이분탐색).

        Args:
            target_net_monthly: 희망 월 실수령액(세후)
            tax_free_monthly: 월 비과세액 (식대 등, 최대 20만원)
            dependents: 부양가족 수 (본인 포함)
            children_8_to_20: 8세~20세 자녀 수
            company_size: 회사 규모

        Returns:
            calculate_from_monthly() 결과에 '역산정보' 키 추가

        Raises:
            ValueError: target_net_monthly가 양수가 아닌 경우
        """
        if target_net_monthly <= 0:
            raise ValueError("희망 실수령액은 양수여야 합니다.")

        low = target_net_monthly
        high = int(target_net_monthly * 2.5)

        best_result = None
        best_diff = float('inf')

        for _ in range(60):
            mid = (low + high) // 2
            result = self.calculate_from_monthly(
                mid, tax_free_monthly, dependents, children_8_to_20, company_size
            )
            net = result['실수령액']['월_실수령액']
            diff = abs(net - target_net_monthly)

            if diff < best_diff:
                best_diff = diff
                best_result = result

            if diff <= 100:
                break
            elif net < target_net_monthly:
                low = mid + 1
            else:
                high = mid - 1

        best_result['역산정보'] = {
            '희망_실수령액': target_net_monthly,
            '필요_세전_월급': best_result['입력정보']['월급여'],
            '필요_세전_연봉': best_result['입력정보']['월급여'] * 12,
            '실제_실수령액': best_result['실수령액']['월_실수령액'],
            '오차': best_result['실수령액']['월_실수령액'] - target_net_monthly,
        }
        return best_result

    def calculate_from_annual(self, annual_salary: int,
                               tax_free_monthly: int = 0,
                               dependents: int = 1,
                               children_8_to_20: int = 0,
                               company_size: str = 'small') -> dict:
        """
        연봉 기준 급여 계산

        Args:
            annual_salary: 연봉
            tax_free_monthly: 월 비과세액 (식대 등, 최대 20만원)
            dependents: 부양가족 수 (본인 포함)
            children_8_to_20: 8세~20세 자녀 수
            company_size: 회사 규모
        """
        monthly_salary = annual_salary // 12
        return self.calculate_from_monthly(
            monthly_salary, tax_free_monthly, dependents,
            children_8_to_20, company_size
        )

    def calculate_from_monthly(self, monthly_salary: int,
                                tax_free_monthly: int = 0,
                                dependents: int = 1,
                                children_8_to_20: int = 0,
                                company_size: str = 'small') -> dict:
        """
        월급 기준 급여 계산
        """
        # 비과세 한도 체크 (식대 최대 20만원)
        tax_free_monthly = min(tax_free_monthly, 200000)

        # 4대보험 계산 (과세급여 기준)
        taxable_salary = monthly_salary - tax_free_monthly

        pension_emp, pension_er = self.calculate_national_pension(taxable_salary)
        health_emp, health_er = self.calculate_health_insurance(taxable_salary)
        care_emp, care_er = self.calculate_long_term_care(health_emp, health_er)
        emp_ins_emp, emp_ins_er = self.calculate_employment_insurance(
            taxable_salary, company_size
        )
        industrial = self.calculate_industrial_accident(taxable_salary)

        # 세금 계산
        income_tax = self.calculate_income_tax(
            monthly_salary, tax_free_monthly, dependents, children_8_to_20
        )
        local_tax = self.calculate_local_income_tax(income_tax)

        # 공제 합계
        total_employee_deduction = (
            pension_emp + health_emp + care_emp + emp_ins_emp +
            income_tax + local_tax
        )

        total_employer_cost = (
            pension_er + health_er + care_er + emp_ins_er + industrial
        )

        # 실수령액
        net_salary = monthly_salary - total_employee_deduction

        return {
            '입력정보': {
                '월급여': monthly_salary,
                '비과세액': tax_free_monthly,
                '과세급여': taxable_salary,
                '부양가족수': dependents,
                '자녀수_8세_20세': children_8_to_20,
                '회사규모': company_size,
            },
            '근로자_공제내역': {
                '국민연금': pension_emp,
                '건강보험': health_emp,
                '장기요양보험': care_emp,
                '고용보험': emp_ins_emp,
                '소득세': income_tax,
                '지방소득세': local_tax,
                '공제합계': total_employee_deduction,
            },
            '회사_부담내역': {
                '국민연금': pension_er,
                '건강보험': health_er,
                '장기요양보험': care_er,
                '고용보험': emp_ins_er,
                '산재보험': industrial,
                '부담합계': total_employer_cost,
            },
            '실수령액': {
                '월_실수령액': net_salary,
                '연_실수령액_추정': net_salary * 12,
            },
            '총_인건비': {
                '월_총인건비': monthly_salary + total_employer_cost,
                '연_총인건비_추정': (monthly_salary + total_employer_cost) * 12,
            }
        }


def format_currency(amount: int) -> str:
    """금액을 원화 형식으로 포맷팅"""
    return f"{amount:,}원"


def print_payslip(result: dict):
    """급여명세서 출력"""
    print("\n" + "=" * 60)
    print("                    급 여 명 세 서")
    print("=" * 60)

    info = result['입력정보']
    print(f"\n【입력 정보】")
    print(f"  월 급여액: {format_currency(info['월급여'])}")
    print(f"  비과세액 (월): {format_currency(info['비과세액'])}")
    print(f"  과세급여: {format_currency(info['과세급여'])}")
    print(f"  부양가족 수: {info['부양가족수']}명")
    print(f"  8~20세 자녀: {info['자녀수_8세_20세']}명")

    emp_ded = result['근로자_공제내역']
    print(f"\n【근로자 공제 내역】")
    print(f"  국민연금:     {format_currency(emp_ded['국민연금']):>15}")
    print(f"  건강보험:     {format_currency(emp_ded['건강보험']):>15}")
    print(f"  장기요양보험: {format_currency(emp_ded['장기요양보험']):>15}")
    print(f"  고용보험:     {format_currency(emp_ded['고용보험']):>15}")
    print(f"  소득세:       {format_currency(emp_ded['소득세']):>15}")
    print(f"  지방소득세:   {format_currency(emp_ded['지방소득세']):>15}")
    print("-" * 40)
    print(f"  공제 합계:    {format_currency(emp_ded['공제합계']):>15}")

    er_cost = result['회사_부담내역']
    print(f"\n【회사 부담 내역】")
    print(f"  국민연금:     {format_currency(er_cost['국민연금']):>15}")
    print(f"  건강보험:     {format_currency(er_cost['건강보험']):>15}")
    print(f"  장기요양보험: {format_currency(er_cost['장기요양보험']):>15}")
    print(f"  고용보험:     {format_currency(er_cost['고용보험']):>15}")
    print(f"  산재보험:     {format_currency(er_cost['산재보험']):>15}")
    print("-" * 40)
    print(f"  부담 합계:    {format_currency(er_cost['부담합계']):>15}")

    net = result['실수령액']
    total = result['총_인건비']
    print(f"\n【계산 결과】")
    print("=" * 40)
    print(f"  월 실수령액:   {format_currency(net['월_실수령액']):>15}")
    print(f"  연 실수령액:   {format_currency(net['연_실수령액_추정']):>15} (추정)")
    print(f"  월 총인건비:   {format_currency(total['월_총인건비']):>15}")
    print(f"  연 총인건비:   {format_currency(total['연_총인건비_추정']):>15} (추정)")
    print("=" * 60)


def main():
    """메인 실행 함수 - 대화형 인터페이스"""
    print("\n" + "=" * 60)
    print("        한국 임금계산기 (2025년 기준)")
    print("=" * 60)

    calc = WageCalculator()

    while True:
        print("\n[급여 유형 선택]")
        print("  1. 연봉으로 계산")
        print("  2. 월급으로 계산")
        print("  q. 종료")

        choice = input("\n선택: ").strip().lower()

        if choice == 'q':
            print("프로그램을 종료합니다.")
            break

        if choice not in ['1', '2']:
            print("잘못된 선택입니다. 다시 선택해주세요.")
            continue

        try:
            if choice == '1':
                salary = int(input("연봉 (원): ").replace(',', ''))
                monthly = salary // 12
            else:
                monthly = int(input("월급 (원): ").replace(',', ''))

            tax_free = input("비과세액/월 (기본 0, 최대 200000): ").strip()
            tax_free = int(tax_free.replace(',', '')) if tax_free else 0

            dependents = input("부양가족 수 (본인 포함, 기본 1): ").strip()
            dependents = int(dependents) if dependents else 1

            children = input("8~20세 자녀 수 (기본 0): ").strip()
            children = int(children) if children else 0

            print("\n회사 규모:")
            print("  1. 150인 미만 (소기업)")
            print("  2. 150~999인 (중기업)")
            print("  3. 1000인 이상 (대기업)")
            size_choice = input("선택 (기본 1): ").strip()

            size_map = {'1': 'small', '2': 'medium', '3': 'large'}
            company_size = size_map.get(size_choice, 'small')

            result = calc.calculate_from_monthly(
                monthly, tax_free, dependents, children, company_size
            )

            print_payslip(result)

        except ValueError:
            print("숫자를 올바르게 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {e}")


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 연봉 5000만원
    print("\n" + "=" * 60)
    print("        [예시] 연봉 5,000만원 계산 결과")
    print("=" * 60)

    calc = WageCalculator()
    result = calc.calculate_from_annual(
        annual_salary=50000000,
        tax_free_monthly=200000,  # 식대 20만원
        dependents=3,             # 본인 + 배우자 + 자녀1
        children_8_to_20=1,       # 자녀 1명
        company_size='small'
    )
    print_payslip(result)

    # 예시 2: 월급 300만원
    print("\n" + "=" * 60)
    print("        [예시] 월급 300만원 계산 결과")
    print("=" * 60)

    result = calc.calculate_from_monthly(
        monthly_salary=3000000,
        tax_free_monthly=100000,
        dependents=1,
        children_8_to_20=0,
        company_size='medium'
    )
    print_payslip(result)

    # 대화형 모드 실행
    # main()
