"""
한국 4대보험료 계산기
- 국민연금, 건강보험, 장기요양보험, 고용보험, 산재보험
- 2026년 기준 요율 적용
- 참고: https://www.nodong.kr/insure_cal
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class CompanySize(Enum):
    """회사 규모 (고용보험 요율 결정용)"""
    UNDER_150 = "150인 미만"
    PRIORITY_SUPPORT = "150인 이상 우선지원대상기업"
    FROM_150_TO_999 = "150인 이상 ~ 1,000인 미만"
    OVER_1000 = "1,000인 이상 / 국가·지자체"


class IndustryType(Enum):
    """산재보험 업종 분류 (29개 업종)"""
    # 업종코드, 업종명, 산재보험료율(%)
    MINING_COAL = ("11", "석탄광업", 18.5)
    MINING_METAL = ("12", "금속 및 비금속광업", 5.8)
    MINING_OTHER = ("13", "기타광업", 8.8)
    FOOD_BEVERAGE = ("21", "식료품제조업", 1.2)
    TEXTILE = ("22", "섬유·의복업", 1.0)
    WOOD_PAPER = ("23", "목재·종이·인쇄업", 1.5)
    CHEMICAL = ("24", "화학·고무·플라스틱제조업", 1.0)
    CEMENT = ("25", "시멘트제조업", 2.1)
    METAL_PRODUCT = ("26", "1차금속·금속제품제조업", 1.5)
    MACHINERY = ("27", "기계기구·금속제품조립제조업", 1.3)
    ELECTRICAL = ("28", "전기기계기구·정밀기구제조업", 0.7)
    SHIPBUILDING = ("29", "선박건조·수리업", 2.6)
    TRANSPORT_EQUIP = ("30", "수송용기계기구제조업", 0.9)
    OTHER_MANUFACTURING = ("31", "기타제조업", 1.2)
    ELECTRICITY_GAS = ("32", "전기·가스·수도사업", 0.7)
    CONSTRUCTION = ("41", "건설업", 3.5)
    TRANSPORT_STORAGE = ("51", "운수·창고·통신업", 1.2)
    FORESTRY = ("61", "임업", 5.0)
    FISHING = ("62", "어업", 2.8)
    AGRICULTURE = ("63", "농업", 2.1)
    OTHERS = ("71", "기타사업", 0.9)
    WHOLESALE_RETAIL = ("72", "도·소매·음식·숙박업", 0.7)
    REAL_ESTATE = ("73", "부동산·임대업", 0.9)
    PROFESSIONAL = ("74", "전문·과학·기술서비스업", 0.5)
    BUSINESS_SERVICE = ("75", "사업서비스업", 0.8)
    EDUCATION = ("76", "교육서비스업", 0.5)
    HEALTH_SOCIAL = ("77", "보건·사회복지사업", 0.7)
    ENTERTAINMENT = ("78", "오락·문화·운동관련사업", 0.8)
    STATE_ORGANIZATION = ("79", "국가·지방자치단체사업", 0.7)
    FINANCE_INSURANCE = ("80", "금융·보험업", 0.5)

    def __init__(self, code: str, name: str, rate: float):
        self._code = code
        self._name = name
        self._rate = rate

    @property
    def code(self) -> str:
        return self._code

    @property
    def industry_name(self) -> str:
        return self._name

    @property
    def rate(self) -> float:
        return self._rate


@dataclass
class InsuranceRates2026:
    """2026년 4대보험 요율 (참고: https://www.4insure.or.kr)"""

    # === 국민연금 ===
    # 총 9.5%, 근로자/사업주 각 4.75%
    national_pension_total: float = 0.095
    national_pension_employee: float = 0.0475
    national_pension_employer: float = 0.0475
    # 기준소득월액 상한/하한 (2025.7.1~2026.6.30)
    national_pension_max_income: int = 6_370_000
    national_pension_min_income: int = 400_000

    # === 건강보험 ===
    # 총 7.19%, 근로자/사업주 각 3.595%
    health_insurance_total: float = 0.0719
    health_insurance_employee: float = 0.03595
    health_insurance_employer: float = 0.03595
    # 보수월액 상한/하한 (2026년)
    health_insurance_max_income: int = 127_725_730
    health_insurance_min_income: int = 280_528
    # 보험료 상한/하한
    health_insurance_max_premium: int = 9_183_460  # 월 상한 (근로자+사업주)
    health_insurance_min_premium: int = 20_160  # 월 하한 (근로자+사업주)

    # === 장기요양보험 ===
    # 건강보험료의 13.14% (장기요양보험료율 0.9448% ÷ 건강보험료율 7.19%)
    long_term_care_rate: float = 0.1314

    # === 고용보험 ===
    # 실업급여: 근로자 0.9%, 사업주 0.9%
    employment_insurance_employee: float = 0.009
    employment_insurance_employer_base: float = 0.009
    # 고용안정·직업능력개발사업 (사업주만 부담, 규모별)
    employment_stability_under_150: float = 0.0025  # 150인 미만
    employment_stability_priority: float = 0.0045  # 우선지원대상기업
    employment_stability_150_to_999: float = 0.0065  # 150~999인
    employment_stability_over_1000: float = 0.0085  # 1000인 이상

    # === 산재보험 ===
    # 전액 사업주 부담
    # 기본 요율: 업종별 상이 (IndustryType enum 참조)
    # 추가 부담금
    commute_accident_rate: float = 0.006  # 출퇴근재해 0.6%
    wage_claim_rate: float = 0.0006  # 임금채권부담금 0.06%
    asbestos_rate: float = 0.0003  # 석면피해구제분담금 0.03%


class InsuranceCalculator:
    """4대보험료 계산기"""

    def __init__(self, rates: Optional[InsuranceRates2026] = None):
        self.rates = rates or InsuranceRates2026()

    def truncate(self, amount: float, unit: int = 10) -> int:
        """금액 절사 (기본 10원 단위)"""
        return int(amount // unit) * unit

    def truncate_1000(self, amount: float) -> int:
        """1000원 미만 절사 (국민연금 기준소득월액용)"""
        return int(amount // 1000) * 1000

    # ==========================================
    # 국민연금 계산
    # ==========================================
    def calc_national_pension(self, monthly_income: int,
                               non_taxable: int = 0) -> dict:
        """
        국민연금 계산

        Args:
            monthly_income: 월 총소득
            non_taxable: 비과세소득

        Returns:
            dict: 근로자부담, 사업주부담, 합계, 기준소득월액
        """
        # 과세소득 = 총소득 - 비과세소득
        taxable_income = monthly_income - non_taxable

        # 기준소득월액 결정 (상한/하한 적용, 1000원 미만 절사)
        base_income = self.truncate_1000(taxable_income)
        base_income = max(self.rates.national_pension_min_income,
                          min(base_income, self.rates.national_pension_max_income))

        # 보험료 계산
        employee = self.truncate(base_income * self.rates.national_pension_employee)
        employer = self.truncate(base_income * self.rates.national_pension_employer)

        return {
            "보험종류": "국민연금",
            "기준소득월액": base_income,
            "요율_근로자": f"{self.rates.national_pension_employee * 100:.2f}%",
            "요율_사업주": f"{self.rates.national_pension_employer * 100:.2f}%",
            "근로자부담": employee,
            "사업주부담": employer,
            "합계": employee + employer,
        }

    # ==========================================
    # 건강보험 계산
    # ==========================================
    def calc_health_insurance(self, monthly_income: int,
                               non_taxable: int = 0) -> dict:
        """
        건강보험 계산

        Args:
            monthly_income: 월 총소득
            non_taxable: 비과세소득

        Returns:
            dict: 근로자부담, 사업주부담, 합계, 보수월액
        """
        # 보수월액 = 총소득 - 비과세소득
        salary_base = monthly_income - non_taxable

        # 상한/하한 적용
        salary_base = max(self.rates.health_insurance_min_income,
                          min(salary_base, self.rates.health_insurance_max_income))

        # 보험료 계산
        employee = self.truncate(salary_base * self.rates.health_insurance_employee)
        employer = self.truncate(salary_base * self.rates.health_insurance_employer)

        # 보험료 상한/하한 적용
        total = employee + employer
        min_each = self.rates.health_insurance_min_premium // 2
        max_each = self.rates.health_insurance_max_premium // 2

        employee = max(min_each, min(employee, max_each))
        employer = max(min_each, min(employer, max_each))

        return {
            "보험종류": "건강보험",
            "보수월액": salary_base,
            "요율_근로자": f"{self.rates.health_insurance_employee * 100:.3f}%",
            "요율_사업주": f"{self.rates.health_insurance_employer * 100:.3f}%",
            "근로자부담": employee,
            "사업주부담": employer,
            "합계": employee + employer,
        }

    # ==========================================
    # 장기요양보험 계산
    # ==========================================
    def calc_long_term_care(self, health_employee: int,
                             health_employer: int) -> dict:
        """
        장기요양보험 계산 (건강보험료 기준)

        Args:
            health_employee: 건강보험 근로자부담액
            health_employer: 건강보험 사업주부담액

        Returns:
            dict: 근로자부담, 사업주부담, 합계
        """
        # 건강보험료의 12.95%
        employee = self.truncate(health_employee * self.rates.long_term_care_rate)
        employer = self.truncate(health_employer * self.rates.long_term_care_rate)

        return {
            "보험종류": "장기요양보험",
            "산정기준": "건강보험료",
            "요율": f"{self.rates.long_term_care_rate * 100:.2f}%",
            "근로자부담": employee,
            "사업주부담": employer,
            "합계": employee + employer,
        }

    # ==========================================
    # 고용보험 계산
    # ==========================================
    def calc_employment_insurance(self, monthly_income: int,
                                   non_taxable: int = 0,
                                   company_size: CompanySize = CompanySize.UNDER_150) -> dict:
        """
        고용보험 계산

        Args:
            monthly_income: 월 총소득
            non_taxable: 비과세소득
            company_size: 회사 규모

        Returns:
            dict: 근로자부담, 사업주부담, 합계
        """
        # 과세소득
        taxable_income = monthly_income - non_taxable

        # 근로자: 실업급여 0.9%
        employee = self.truncate(taxable_income * self.rates.employment_insurance_employee)

        # 사업주: 실업급여 0.9% + 고용안정·직업능력개발사업 (규모별)
        employer_base = self.truncate(taxable_income * self.rates.employment_insurance_employer_base)

        # 규모별 고용안정사업 요율
        if company_size == CompanySize.UNDER_150:
            stability_rate = self.rates.employment_stability_under_150
        elif company_size == CompanySize.PRIORITY_SUPPORT:
            stability_rate = self.rates.employment_stability_priority
        elif company_size == CompanySize.FROM_150_TO_999:
            stability_rate = self.rates.employment_stability_150_to_999
        else:
            stability_rate = self.rates.employment_stability_over_1000

        employer_stability = self.truncate(taxable_income * stability_rate)
        employer_total = employer_base + employer_stability

        return {
            "보험종류": "고용보험",
            "과세소득": taxable_income,
            "회사규모": company_size.value,
            "요율_근로자_실업급여": f"{self.rates.employment_insurance_employee * 100:.2f}%",
            "요율_사업주_실업급여": f"{self.rates.employment_insurance_employer_base * 100:.2f}%",
            "요율_사업주_고용안정": f"{stability_rate * 100:.2f}%",
            "근로자부담": employee,
            "사업주부담_실업급여": employer_base,
            "사업주부담_고용안정": employer_stability,
            "사업주부담": employer_total,
            "합계": employee + employer_total,
        }

    # ==========================================
    # 산재보험 계산
    # ==========================================
    def calc_industrial_accident(self, monthly_income: int,
                                  non_taxable: int = 0,
                                  industry: IndustryType = IndustryType.OTHERS) -> dict:
        """
        산재보험 계산 (전액 사업주 부담)

        Args:
            monthly_income: 월 총소득 (월평균보수)
            non_taxable: 비과세소득
            industry: 업종

        Returns:
            dict: 사업주부담 (업종별 요율 + 부가금)
        """
        # 월평균보수 = 총소득 - 비과세소득
        avg_salary = monthly_income - non_taxable

        # 업종별 산재보험료율
        industry_rate = industry.rate / 100

        # 총 요율 = 업종요율 + 출퇴근재해 + 임금채권부담금 + 석면피해구제분담금
        total_rate = (industry_rate +
                      self.rates.commute_accident_rate +
                      self.rates.wage_claim_rate +
                      self.rates.asbestos_rate)

        # 각 항목별 계산
        industry_premium = self.truncate(avg_salary * industry_rate)
        commute_premium = self.truncate(avg_salary * self.rates.commute_accident_rate)
        wage_claim = self.truncate(avg_salary * self.rates.wage_claim_rate)
        asbestos = self.truncate(avg_salary * self.rates.asbestos_rate)

        employer_total = industry_premium + commute_premium + wage_claim + asbestos

        return {
            "보험종류": "산재보험",
            "월평균보수": avg_salary,
            "업종": industry.industry_name,
            "업종코드": industry.code,
            "요율_업종별": f"{industry.rate:.2f}%",
            "요율_출퇴근재해": f"{self.rates.commute_accident_rate * 100:.2f}%",
            "요율_임금채권부담금": f"{self.rates.wage_claim_rate * 100:.2f}%",
            "요율_석면피해구제": f"{self.rates.asbestos_rate * 100:.2f}%",
            "요율_합계": f"{total_rate * 100:.2f}%",
            "보험료_업종별": industry_premium,
            "보험료_출퇴근재해": commute_premium,
            "보험료_임금채권부담금": wage_claim,
            "보험료_석면피해구제": asbestos,
            "근로자부담": 0,
            "사업주부담": employer_total,
            "합계": employer_total,
        }

    # ==========================================
    # 4대보험 통합 계산
    # ==========================================
    def calculate_all(self, monthly_income: int,
                       non_taxable: int = 0,
                       company_size: CompanySize = CompanySize.UNDER_150,
                       industry: IndustryType = IndustryType.OTHERS) -> dict:
        """
        4대보험 통합 계산

        Args:
            monthly_income: 월 총소득
            non_taxable: 비과세소득
            company_size: 회사 규모
            industry: 업종 (산재보험용)

        Returns:
            dict: 모든 보험료 상세 내역
        """
        # 각 보험 계산
        pension = self.calc_national_pension(monthly_income, non_taxable)
        health = self.calc_health_insurance(monthly_income, non_taxable)
        care = self.calc_long_term_care(health["근로자부담"], health["사업주부담"])
        employment = self.calc_employment_insurance(monthly_income, non_taxable, company_size)
        accident = self.calc_industrial_accident(monthly_income, non_taxable, industry)

        # 합계 계산
        total_employee = (pension["근로자부담"] +
                          health["근로자부담"] +
                          care["근로자부담"] +
                          employment["근로자부담"])

        total_employer = (pension["사업주부담"] +
                          health["사업주부담"] +
                          care["사업주부담"] +
                          employment["사업주부담"] +
                          accident["사업주부담"])

        return {
            "입력정보": {
                "월소득": monthly_income,
                "비과세소득": non_taxable,
                "과세소득": monthly_income - non_taxable,
                "회사규모": company_size.value,
                "업종": industry.industry_name,
            },
            "국민연금": pension,
            "건강보험": health,
            "장기요양보험": care,
            "고용보험": employment,
            "산재보험": accident,
            "합계": {
                "근로자부담_합계": total_employee,
                "사업주부담_합계": total_employer,
                "총합계": total_employee + total_employer,
            }
        }


def format_currency(amount: int) -> str:
    """금액을 원화 형식으로 포맷팅"""
    return f"{amount:,}원"


def print_insurance_detail(result: dict):
    """4대보험료 상세 내역 출력"""
    print("\n" + "=" * 70)
    print("                    4 대 보 험 료   계 산 서")
    print("=" * 70)

    # 입력 정보
    info = result["입력정보"]
    print(f"\n【입력 정보】")
    print(f"  월 소득:      {format_currency(info['월소득'])}")
    print(f"  비과세소득:   {format_currency(info['비과세소득'])}")
    print(f"  과세소득:     {format_currency(info['과세소득'])}")
    print(f"  회사 규모:    {info['회사규모']}")
    print(f"  업종:         {info['업종']}")

    # 국민연금
    pension = result["국민연금"]
    print(f"\n【국민연금】 요율: {pension['요율_근로자']} / {pension['요율_사업주']}")
    print(f"  기준소득월액:   {format_currency(pension['기준소득월액'])}")
    print(f"  근로자 부담:    {format_currency(pension['근로자부담']):>15}")
    print(f"  사업주 부담:    {format_currency(pension['사업주부담']):>15}")
    print(f"  합계:           {format_currency(pension['합계']):>15}")

    # 건강보험
    health = result["건강보험"]
    print(f"\n【건강보험】 요율: {health['요율_근로자']} / {health['요율_사업주']}")
    print(f"  보수월액:       {format_currency(health['보수월액'])}")
    print(f"  근로자 부담:    {format_currency(health['근로자부담']):>15}")
    print(f"  사업주 부담:    {format_currency(health['사업주부담']):>15}")
    print(f"  합계:           {format_currency(health['합계']):>15}")

    # 장기요양보험
    care = result["장기요양보험"]
    print(f"\n【장기요양보험】 요율: 건강보험료의 {care['요율']}")
    print(f"  근로자 부담:    {format_currency(care['근로자부담']):>15}")
    print(f"  사업주 부담:    {format_currency(care['사업주부담']):>15}")
    print(f"  합계:           {format_currency(care['합계']):>15}")

    # 고용보험
    emp = result["고용보험"]
    print(f"\n【고용보험】 회사규모: {emp['회사규모']}")
    print(f"  요율 (근로자 - 실업급여):      {emp['요율_근로자_실업급여']}")
    print(f"  요율 (사업주 - 실업급여):      {emp['요율_사업주_실업급여']}")
    print(f"  요율 (사업주 - 고용안정):      {emp['요율_사업주_고용안정']}")
    print(f"  근로자 부담:    {format_currency(emp['근로자부담']):>15}")
    print(f"  사업주 부담:    {format_currency(emp['사업주부담']):>15}")
    print(f"    - 실업급여:   {format_currency(emp['사업주부담_실업급여']):>15}")
    print(f"    - 고용안정:   {format_currency(emp['사업주부담_고용안정']):>15}")
    print(f"  합계:           {format_currency(emp['합계']):>15}")

    # 산재보험
    acc = result["산재보험"]
    print(f"\n【산재보험】 업종: {acc['업종']} (코드: {acc['업종코드']})")
    print(f"  요율 (업종별):          {acc['요율_업종별']}")
    print(f"  요율 (출퇴근재해):      {acc['요율_출퇴근재해']}")
    print(f"  요율 (임금채권부담금):  {acc['요율_임금채권부담금']}")
    print(f"  요율 (석면피해구제):    {acc['요율_석면피해구제']}")
    print(f"  요율 합계:              {acc['요율_합계']}")
    print(f"  근로자 부담:    {format_currency(acc['근로자부담']):>15} (전액 사업주 부담)")
    print(f"  사업주 부담:    {format_currency(acc['사업주부담']):>15}")
    print(f"    - 업종별:     {format_currency(acc['보험료_업종별']):>15}")
    print(f"    - 출퇴근재해: {format_currency(acc['보험료_출퇴근재해']):>15}")
    print(f"    - 임금채권:   {format_currency(acc['보험료_임금채권부담금']):>15}")
    print(f"    - 석면피해:   {format_currency(acc['보험료_석면피해구제']):>15}")

    # 총합계
    total = result["합계"]
    print("\n" + "-" * 70)
    print("【4대보험료 총합계】")
    print(f"  근로자 부담 합계:  {format_currency(total['근로자부담_합계']):>18}")
    print(f"  사업주 부담 합계:  {format_currency(total['사업주부담_합계']):>18}")
    print("=" * 70)
    print(f"  총 합계:           {format_currency(total['총합계']):>18}")
    print("=" * 70)


def print_industry_list():
    """업종 목록 출력"""
    print("\n【산재보험 업종별 요율 (2025년)】")
    print("-" * 50)
    print(f"{'코드':<6}{'업종명':<30}{'요율':>10}")
    print("-" * 50)
    for industry in IndustryType:
        print(f"{industry.code:<6}{industry.industry_name:<30}{industry.rate:>8.2f}%")
    print("-" * 50)


def main():
    """대화형 인터페이스"""
    print("\n" + "=" * 70)
    print("          한국 4대보험료 계산기 (2025년 기준)")
    print("=" * 70)

    calc = InsuranceCalculator()

    while True:
        print("\n[메뉴 선택]")
        print("  1. 4대보험료 계산")
        print("  2. 업종별 산재보험 요율 보기")
        print("  q. 종료")

        choice = input("\n선택: ").strip().lower()

        if choice == 'q':
            print("프로그램을 종료합니다.")
            break

        if choice == '2':
            print_industry_list()
            continue

        if choice != '1':
            print("잘못된 선택입니다.")
            continue

        try:
            # 월소득 입력
            income = input("\n월 소득 (원): ").replace(',', '')
            monthly_income = int(income)

            # 비과세소득 입력
            non_tax = input("비과세소득 (기본 0): ").strip().replace(',', '')
            non_taxable = int(non_tax) if non_tax else 0

            # 회사 규모 선택
            print("\n[회사 규모 선택]")
            print("  1. 150인 미만")
            print("  2. 150인 이상 우선지원대상기업")
            print("  3. 150인 이상 ~ 1,000인 미만")
            print("  4. 1,000인 이상 / 국가·지자체")
            size_choice = input("선택 (기본 1): ").strip()

            size_map = {
                '1': CompanySize.UNDER_150,
                '2': CompanySize.PRIORITY_SUPPORT,
                '3': CompanySize.FROM_150_TO_999,
                '4': CompanySize.OVER_1000,
            }
            company_size = size_map.get(size_choice, CompanySize.UNDER_150)

            # 업종 선택
            print("\n[업종 선택] (산재보험용)")
            print("  주요 업종:")
            print("  21. 기타사업 (일반 사무직)")
            print("  22. 도·소매·음식·숙박업")
            print("  23. 전문·과학·기술서비스업")
            print("  24. 금융·보험업")
            print("  25. 건설업")
            print("  26. 제조업 (기계기구)")
            print("  0. 업종 목록 보기")
            industry_choice = input("업종 코드 입력 (기본 71-기타사업): ").strip()

            if industry_choice == '0':
                print_industry_list()
                industry_choice = input("업종 코드 입력: ").strip()

            # 업종 찾기
            industry = IndustryType.OTHERS
            for ind in IndustryType:
                if ind.code == industry_choice:
                    industry = ind
                    break

            # 계산 및 출력
            result = calc.calculate_all(monthly_income, non_taxable, company_size, industry)
            print_insurance_detail(result)

        except ValueError:
            print("숫자를 올바르게 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {e}")


# 실행
if __name__ == "__main__":
    # 예시 계산
    calc = InsuranceCalculator()

    print("\n" + "=" * 70)
    print("    [예시 1] 월급 300만원, 비과세 20만원, 150인 미만, 기타사업")
    print("=" * 70)
    result = calc.calculate_all(
        monthly_income=3_000_000,
        non_taxable=200_000,
        company_size=CompanySize.UNDER_150,
        industry=IndustryType.OTHERS
    )
    print_insurance_detail(result)

    print("\n" + "=" * 70)
    print("    [예시 2] 월급 500만원, 비과세 20만원, 1000인 이상, 금융보험업")
    print("=" * 70)
    result = calc.calculate_all(
        monthly_income=5_000_000,
        non_taxable=200_000,
        company_size=CompanySize.OVER_1000,
        industry=IndustryType.FINANCE_INSURANCE
    )
    print_insurance_detail(result)

    # 대화형 모드
    # main()
