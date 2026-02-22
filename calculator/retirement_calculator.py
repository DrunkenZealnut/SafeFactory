"""퇴직금 계산기 - 고용노동부 퇴직금 계산 공식 기반.

Reference: https://www.moel.go.kr/retirementpayCal.do

퇴직금 = 1일 평균임금 x 30(일) x (재직일수 / 365)
"""

import calendar
from datetime import date


def _subtract_months(dt: date, months: int) -> date:
    """Subtract *months* from *dt*, clamping day to the last valid day."""
    month = dt.month - months
    year = dt.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class RetirementPayCalculator:
    """고용노동부 퇴직금 계산기.

    Parameters
    ----------
    start_date : str
        입사일자 ``"YYYY-MM-DD"``.
    end_date : str
        퇴직일자 ``"YYYY-MM-DD"`` (마지막 근무일의 **다음 날**).
    monthly_basic_pay : list[int] | int
        퇴직 전 3개월 기본급.  ``int`` 이면 3개월 동일 적용.
    monthly_other_pay : list[int] | int
        퇴직 전 3개월 기타수당.  ``int`` 이면 3개월 동일 적용.
    annual_bonus : int
        연간 상여금 총액 (기본값 0).
    annual_leave_pay : int
        연차수당 (기본값 0).
    excluded_days_avg : int
        미산입기간 일수 — 평균임금 산정기간에서 제외 (기본값 0).
    excluded_days_service : int
        근무제외기간 일수 — 근속기간에서 제외 (기본값 0).
    ordinary_daily_wage : int | None
        1일 통상임금.  제공 시 평균임금과 비교하여 큰 값 적용.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        monthly_basic_pay,
        monthly_other_pay=0,
        annual_bonus: int = 0,
        annual_leave_pay: int = 0,
        excluded_days_avg: int = 0,
        excluded_days_service: int = 0,
        ordinary_daily_wage: int | None = None,
    ):
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)

        # 3개월 기본급/기타수당을 리스트로 정규화
        if isinstance(monthly_basic_pay, (int, float)):
            self.basic_pay = [int(monthly_basic_pay)] * 3
        else:
            if len(monthly_basic_pay) != 3:
                raise ValueError("monthly_basic_pay must have exactly 3 elements")
            self.basic_pay = [int(x) for x in monthly_basic_pay]

        if isinstance(monthly_other_pay, (int, float)):
            self.other_pay = [int(monthly_other_pay)] * 3
        else:
            if len(monthly_other_pay) != 3:
                raise ValueError("monthly_other_pay must have exactly 3 elements")
            self.other_pay = [int(x) for x in monthly_other_pay]

        self.annual_bonus = int(annual_bonus)
        self.annual_leave_pay = int(annual_leave_pay)
        self.excluded_days_avg = int(excluded_days_avg)
        self.excluded_days_service = int(excluded_days_service)
        self.ordinary_daily_wage = int(ordinary_daily_wage) if ordinary_daily_wage else None

        # 3개월 산정기간 사전 계산 (validate/calculate 양쪽에서 사용)
        self._three_months_ago = _subtract_months(self.end_date, 3)
        self._total_days_3m = (self.end_date - self._three_months_ago).days

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def calculate(self) -> dict:
        """퇴직금 계산 결과를 dict 로 반환."""
        self._validate()

        # 1) 재직일수
        total_service_days = (self.end_date - self.start_date).days
        service_days = total_service_days - self.excluded_days_service

        # 2) 퇴직 전 3개월 총일수
        total_days_3m = self._total_days_3m
        calc_days_3m = total_days_3m - self.excluded_days_avg

        # 3) 3개월 임금총액
        wage_3m = sum(self.basic_pay) + sum(self.other_pay)

        # 4) 상여금 가산액 = 연간상여금 x 3/12
        bonus_addition = round(self.annual_bonus * 3 / 12)

        # 5) 연차수당 가산액 = 연차수당 x 3/12
        leave_addition = round(self.annual_leave_pay * 3 / 12)

        # 6) 평균임금 산정 기초금액
        total_wage_base = wage_3m + bonus_addition + leave_addition

        # 7) 1일 평균임금
        daily_avg_wage = round(total_wage_base / calc_days_3m) if calc_days_3m > 0 else 0

        # 8) 통상임금 비교 — 통상임금이 더 크면 통상임금 적용
        applied_daily_wage = daily_avg_wage
        used_ordinary = False
        if self.ordinary_daily_wage and self.ordinary_daily_wage > daily_avg_wage:
            applied_daily_wage = self.ordinary_daily_wage
            used_ordinary = True

        # 9) 퇴직금 = 1일 평균임금 x 30 x (재직일수 / 365)
        retirement_pay = round(applied_daily_wage * 30 * (service_days / 365))

        return {
            '입력정보': {
                '입사일자': self.start_date.isoformat(),
                '퇴직일자': self.end_date.isoformat(),
                '재직일수_전체': total_service_days,
                '근무제외기간': self.excluded_days_service,
                '재직일수': service_days,
            },
            '평균임금_산정': {
                '퇴직전_3개월_총일수': total_days_3m,
                '미산입기간': self.excluded_days_avg,
                '산정기간_일수': calc_days_3m,
                '기본급_3개월': self.basic_pay,
                '기타수당_3개월': self.other_pay,
                '임금총액_3개월': wage_3m,
                '상여금_가산액': bonus_addition,
                '연차수당_가산액': leave_addition,
                '평균임금_기초금액': total_wage_base,
                '1일_평균임금': daily_avg_wage,
            },
            '퇴직금_산출': {
                '1일_평균임금': daily_avg_wage,
                '1일_통상임금': self.ordinary_daily_wage,
                '통상임금_적용여부': used_ordinary,
                '적용_1일_임금': applied_daily_wage,
                '재직일수': service_days,
                '퇴직금': retirement_pay,
            },
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        if self.end_date <= self.start_date:
            raise ValueError("퇴직일자는 입사일자보다 이후여야 합니다.")

        total_days = (self.end_date - self.start_date).days
        if total_days < 365:
            raise ValueError(
                f"재직일수가 365일 미만({total_days}일)입니다. "
                "퇴직금은 1년 이상 근무한 경우에만 발생합니다."
            )

        if any(p < 0 for p in self.basic_pay):
            raise ValueError("기본급은 음수일 수 없습니다.")

        if any(p < 0 for p in self.other_pay):
            raise ValueError("기타수당은 음수일 수 없습니다.")

        if self.excluded_days_avg >= self._total_days_3m:
            raise ValueError(
                f"미산입기간({self.excluded_days_avg}일)이 "
                f"3개월 산정기간({self._total_days_3m}일) 이상입니다."
            )


# -----------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------
if __name__ == '__main__':
    calc = RetirementPayCalculator(
        start_date='2020-01-01',
        end_date='2025-01-01',
        monthly_basic_pay=3_000_000,
        monthly_other_pay=500_000,
        annual_bonus=6_000_000,
        annual_leave_pay=1_200_000,
    )
    result = calc.calculate()

    print("=== 퇴직금 계산 결과 ===")
    for section, data in result.items():
        print(f"\n[{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {v}")
