"""연차유급휴가 계산기 - 근로기준법 제60조 기준.

Reference: https://labor.moel.go.kr/cmmt/calAnnlVctn.do

근로기준법 제60조:
  제1항: 1년간 80% 이상 출근 시 15일의 유급휴가
  제2항: 1년 미만 근로자에게 1개월 개근 시 1일 유급휴가 (최대 11일)
  제3항: 제2항 사용분은 제1항 휴가에서 공제하지 않음 (2020.3.31 개정)
  제4항: 3년 이상 계속 근로 시 최초 1년 초과 매 2년에 1일 가산 (총 25일 한도)
"""

import calendar
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Date helpers (standard library only)
# ---------------------------------------------------------------------------

def _add_months(dt: date, months: int) -> date:
    """Add *months* to *dt*, clamping day to the last valid day."""
    total_months = dt.month + months
    year = dt.year + (total_months - 1) // 12
    month = (total_months - 1) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _add_years(dt: date, years: int) -> date:
    """Add *years* to *dt*, clamping Feb 29 → Feb 28 in non-leap years."""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        return dt.replace(year=dt.year + years, day=28)


def _count_completed_months(start: date, end: date) -> int:
    """Count completed months between *start* and *end*, max 11."""
    for m in range(11, -1, -1):
        if _add_months(start, m) <= end:
            return m
    return 0


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

# 2020.3.31 개정: 1년 미만 월차 사용분 비차감
_DEDUCTION_CUTOFF = date(2020, 3, 31)


def annual_leave_for_year(completed_years: int) -> int:
    """Return statutory annual leave days for *completed_years* of service.

    - completed_years == 0: handled separately (monthly leave)
    - completed_years >= 1: 15 + max(0, (completed_years - 1) // 2), cap 25
    """
    if completed_years < 1:
        return 0
    return min(25, 15 + max(0, (completed_years - 1) // 2))


class AnnualLeaveCalculator:
    """연차유급휴가 계산기.

    Parameters
    ----------
    hire_date : str
        입사일자 ``"YYYY-MM-DD"``.
    end_date : str | None
        퇴직일자 또는 기준일자 ``"YYYY-MM-DD"``.
        ``None`` 이면 오늘 날짜 사용.
    """

    def __init__(self, hire_date: str, end_date: str | None = None):
        self.hire_date = date.fromisoformat(hire_date)
        self.end_date = date.fromisoformat(end_date) if end_date else date.today()

    def calculate(self) -> dict:
        """연차 발생 내역을 연도별로 계산하여 반환."""
        self._validate()

        yearly = []
        total_leave = 0
        first_anniversary = _add_years(self.hire_date, 1)

        # ── 1년차: 월차 (제60조 제2항) ──────────────────────────
        if self.end_date < first_anniversary:
            # 아직 1년 미만
            months = _count_completed_months(self.hire_date, self.end_date)
            monthly_leave = min(months, 11)
            yearly.append({
                '근무년차': 1,
                '기간_시작': self.hire_date.isoformat(),
                '기간_종료': self.end_date.isoformat(),
                '유형': '월차 (제60조 제2항)',
                '발생일수': monthly_leave,
                '비고': f'{months}개월 개근 시',
            })
            total_leave += monthly_leave
        else:
            # 1년차 완료
            monthly_leave = 11
            yearly.append({
                '근무년차': 1,
                '기간_시작': self.hire_date.isoformat(),
                '기간_종료': (first_anniversary - timedelta(days=1)).isoformat(),
                '유형': '월차 (제60조 제2항)',
                '발생일수': monthly_leave,
                '비고': '매월 1일씩 (최대 11일)',
            })
            total_leave += monthly_leave

            # 차감 여부 (2020.3.31 이전 입사 시)
            deduction_applies = self.hire_date < _DEDUCTION_CUTOFF

            # ── 2년차 이후: 연차 (제60조 제1항, 제4항) ──────────
            completed_years = 1
            while True:
                period_start = _add_years(self.hire_date, completed_years)
                if period_start > self.end_date:
                    break

                next_anniversary = _add_years(self.hire_date, completed_years + 1)
                actual_end = min(
                    next_anniversary - timedelta(days=1),
                    self.end_date - timedelta(days=1),
                )

                days = annual_leave_for_year(completed_years)

                # 2년차(completed_years==1)에서 차감 적용
                note_parts = []
                if completed_years == 1 and deduction_applies:
                    note_parts.append(
                        '※ 2020.3.31 이전 입사: 1년차 사용 월차분 차감 가능'
                    )

                bonus = max(0, (completed_years - 1) // 2)
                if bonus > 0:
                    note_parts.append(f'기본 15일 + 가산 {bonus}일 (제60조 제4항)')
                else:
                    note_parts.append('제60조 제1항')

                yearly.append({
                    '근무년차': completed_years + 1,
                    '기간_시작': period_start.isoformat(),
                    '기간_종료': actual_end.isoformat(),
                    '유형': '연차',
                    '발생일수': days,
                    '비고': ', '.join(note_parts),
                })
                total_leave += days

                completed_years += 1
                if next_anniversary >= self.end_date:
                    break

        total_service_days = (self.end_date - self.hire_date).days
        completed_years_total = 0
        check = self.hire_date
        while _add_years(check, 1) <= self.end_date:
            completed_years_total += 1
            check = _add_years(self.hire_date, completed_years_total)

        return {
            '입력정보': {
                '입사일자': self.hire_date.isoformat(),
                '기준일자': self.end_date.isoformat(),
                '총_재직일수': total_service_days,
                '만_근속연수': completed_years_total,
            },
            '연도별_내역': yearly,
            '총_발생_연차일수': total_leave,
        }

    def _validate(self):
        if self.end_date <= self.hire_date:
            raise ValueError("기준일자는 입사일자보다 이후여야 합니다.")


# -----------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------
if __name__ == '__main__':
    # 2020.06.01 입사, 현재 기준
    calc = AnnualLeaveCalculator(
        hire_date='2020-06-01',
        end_date='2026-02-21',
    )
    result = calc.calculate()

    info = result['입력정보']
    print(f"=== 연차휴가 계산 결과 ===")
    print(f"입사일: {info['입사일자']}, 기준일: {info['기준일자']}")
    print(f"재직일수: {info['총_재직일수']}일, 만 근속: {info['만_근속연수']}년")
    print()

    print(f"{'년차':>4} | {'유형':^20} | {'일수':>4} | 비고")
    print("-" * 70)
    for y in result['연도별_내역']:
        print(f"{y['근무년차']:>4} | {y['유형']:^20} | {y['발생일수']:>3}일 | {y['비고']}")
    print("-" * 70)
    print(f"{'합계':>4} |{' ':^21}| {result['총_발생_연차일수']:>3}일 |")
