from datetime import datetime, time

class MarketCalendar:
    def is_market_open(self, ts: datetime) -> bool:
        # naive US equities example (M-F, 9:30-16:00 local)
        if ts.weekday() >= 5: return False
        t = ts.time()
        return time(9,30) <= t <= time(16,0)
