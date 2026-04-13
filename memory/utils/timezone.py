"""
时区工具

统一管理时间处理，避免 naive datetime 和 aware datetime 混用。
默认使用 UTC 时区。
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import re


# 默认时区
DEFAULT_TIMEZONE = timezone.utc


def now() -> datetime:
    """
    获取当前时间（UTC，时区感知）
    
    Returns:
        当前 UTC 时间
    """
    return datetime.now(timezone.utc)


def from_naive(dt: datetime, tz: Optional[timezone] = None) -> datetime:
    """
    将 naive datetime 转换为 aware datetime
    
    Args:
        dt: naive datetime
        tz: 目标时区，默认 UTC
    
    Returns:
        aware datetime
    """
    if tz is None:
        tz = DEFAULT_TIMEZONE
    
    if dt.tzinfo is not None:
        return dt  # 已经是 aware
    
    return dt.replace(tzinfo=tz)


def to_naive(dt: datetime) -> datetime:
    """
    将 aware datetime 转换为 naive datetime（移除时区信息）
    
    Args:
        dt: aware datetime
    
    Returns:
        naive datetime
    """
    if dt.tzinfo is None:
        return dt  # 已经是 naive
    
    return dt.replace(tzinfo=None)


def to_utc(dt: datetime) -> datetime:
    """
    将 datetime 转换为 UTC
    
    Args:
        dt: datetime
    
    Returns:
        UTC 时区的 datetime
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=DEFAULT_TIMEZONE)
    return dt.astimezone(DEFAULT_TIMEZONE)


def parse_datetime(dt_str: str) -> datetime:
    """
    解析字符串为 datetime（支持多种格式）
    
    Args:
        dt_str: 日期时间字符串
    
    Returns:
        aware datetime (UTC)
    
    Raises:
        ValueError: 无法解析
    """
    # 尝试常见格式
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%d',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%dT%H:%M:%S',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return from_naive(dt)
        except ValueError:
            continue
    
    raise ValueError(f"无法解析日期时间: {dt_str}")


def days_diff(dt1: datetime, dt2: datetime) -> float:
    """
    计算两个 datetime 之间的天数差
    
    Args:
        dt1: 第一个时间
        dt2: 第二个时间
    
    Returns:
        天数差（可为负）
    """
    # 确保都是 aware
    dt1_aware = from_naive(dt1)
    dt2_aware = from_naive(dt2)
    
    delta = dt2_aware - dt1_aware
    return delta.total_seconds() / (24 * 3600)


def ensure_aware(dt: Optional[datetime], reference: Optional[datetime] = None) -> datetime:
    """
    确保 datetime 是时区感知的
    
    如果输入是 None，使用 reference 或当前时间。
    如果输入是 naive datetime，转换为 UTC aware。
    如果输入是 aware datetime，直接返回。
    
    Args:
        dt: 输入 datetime
        reference: 参考时间（当 dt 为 None 时使用）
    
    Returns:
        aware datetime
    """
    if dt is None:
        return reference if reference is not None else now()
    
    return from_naive(dt)


__all__ = [
    'now', 'from_naive', 'to_naive', 'to_utc',
    'parse_datetime', 'days_diff', 'ensure_aware',
    'DEFAULT_TIMEZONE',
]
