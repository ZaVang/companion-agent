"""
因果推断配置
"""

from typing import Tuple, Optional
from pydantic import BaseModel, Field


class CausalConfig(BaseModel):
    """因果推断配置"""
    # 时间窗口
    temporal_window_ms: int = 5000  # 5秒内视为相关
    
    # 因果判定阈值
    min_cooccurrence_count: int = 3   # 最小共现次数
    min_causal_strength: float = 0.6 # 最小因果强度
    
    # 统计参数
    confidence_decay: float = 0.9    # 置信度衰减因子
    max_chain_length: int = 10       # 最大因果链长度
    
    # 预测参数
    prediction_confidence_threshold: float = 0.5  # 预测置信度阈值
    max_predictions: int = 5                  # 最大预测数量
    
    # 性能参数
    batch_size: int = 1000  # 批处理大小


# 默认配置
DEFAULT_CAUSAL_CONFIG = CausalConfig()
