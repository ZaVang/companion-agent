from typing import Union
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class TimeModule:
    def __init__(self, timezone: str = "UTC"):
        self.timezone = ZoneInfo(timezone)

    def get_current_time(self, is_str=True) -> Union[str, datetime]:
        """获取当前时间并返回易读格式"""
        localized_time = datetime.now(self.timezone)
        if is_str:
            readable_time = localized_time.strftime('%Y-%m-%d %H:%M:%S')
            return readable_time
        else:
            return localized_time

    def time_difference(self, start_time: datetime, end_time: datetime) -> timedelta:
        """计算两个时间点之间的差异"""
        return end_time - start_time

    def add_days(self, date: datetime, days: int) -> datetime:
        """给定日期增加或减少天数"""
        return date + timedelta(days=days)

    def to_timestamp(self, date: datetime) -> int:
        """将日期转换为UNIX时间戳"""
        return int(date.timestamp())

    def from_timestamp(self, timestamp: int) -> datetime:
        """将UNIX时间戳转换为日期"""
        return datetime.fromtimestamp(timestamp, tz=self.timezone)

    def get_specific_time(self, day_offset: int = 0, hour: int = 0, minute: int = 0) -> datetime:
        """获取特定天的特定时刻，例如获取今天午夜或明天的中午"""
        now = datetime.now(self.timezone)
        target_date = now + timedelta(days=day_offset)
        return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
