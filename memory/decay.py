"""
动态衰减系统模块

实现记忆的动态衰减机制，模拟神经元的自然遗忘过程。

核心概念:
- 不同 event_type 有不同的基础衰减率
- 冲击力(impact_score) 越高，衰减越慢
- 衰减是可叠加的，时间越长影响越大
"""

import math
from typing import Dict, List, Literal, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import uuid


# ============== 配置常量 ==============

# 衰减率配置
DECAY_RATE_RANGE = (0.98, 0.9999)  # 快衰减 ~ 极慢衰减

# event_type 基础衰减率（每天）
BASE_DECAY_RATES: Dict[str, float] = {
    'chat': 0.995,           # 对话衰减较快（日常闲聊）
    'perception': 0.990,     # 感知记忆衰减中等
    'thought': 0.992,         # 思考记忆衰减较慢
    'reflection': 0.998,     # 反思记忆衰减很慢
    'experience': 0.985,     # 体验记忆衰减较快
}

# 冲击力对衰减率的影响系数
IMPACT_DECAY_FACTOR: float = 0.1  # impact_score * factor * decay_rate


# ============== 数据模型 ==============

class DecayConfig(BaseModel):
    """衰减系统配置"""
    decay_rate_range: Tuple[float, float] = DECAY_RATE_RANGE
    base_decay_rates: Dict[str, float] = Field(default_factory=lambda: BASE_DECAY_RATES.copy())
    impact_decay_factor: float = IMPACT_DECAY_FACTOR


class NeuronDecayState(BaseModel):
    """神经元的衰减状态"""
    event_id: uuid.UUID
    event_type: str
    base_decay_rate: float = 0.995  # 基础衰减率
    impact_score: float = 0.5        # 冲击力评分 [0, 1]
    decay_rate: float = 0.995        # 当前实际衰减率
    created_at: datetime = Field(default_factory=datetime.now)
    last_decay_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_adjusted_decay_rate(self, config: Optional[DecayConfig] = None) -> float:
        """
        计算调整后的衰减率
        
        公式: adjusted_rate = base_rate + impact_score * factor * (1 - base_rate)
        
        设计原理:
        - 高冲击力记忆的衰减率接近 1（极慢衰减）
        - 低冲击力记忆的衰减率较低（快衰减）
        """
        if config is None:
            config = DecayConfig()
        
        base = self.base_decay_rate
        impact_factor = self.impact_score * config.impact_decay_factor
        
        # 调整公式：高冲击力接近 1，低冲击力保持 base
        adjusted = base + impact_factor * (1 - base)
        
        # 限制范围
        min_rate, max_rate = config.decay_rate_range
        return max(min_rate, min(max_rate, adjusted))
    
    def time_elapsed_days(self, reference_time: Optional[datetime] = None) -> float:
        """计算自上次衰减以来的天数"""
        if reference_time is None:
            reference_time = datetime.now()
        delta = reference_time - self.last_decay_at
        return delta.total_seconds() / (24 * 3600)


# ============== 核心算法 ==============

def calculate_decay_rate(
    event_type: str,
    impact_score: float,
    config: Optional[DecayConfig] = None
) -> float:
    """
    计算单个神经元的衰减率
    
    这是暴露给外部的主要接口函数。
    
    Args:
        event_type: 事件类型 ('chat', 'perception', 'thought', 'reflection', 'experience')
        impact_score: 冲击力评分 [0, 1]
        config: 衰减配置
    
    Returns:
        调整后的衰减率 [0.98, 0.9999]
    """
    if config is None:
        config = DecayConfig()
    
    # 获取基础衰减率
    base_rate = config.base_decay_rates.get(event_type, 0.995)
    
    # 计算冲击力调整
    impact_factor = impact_score * config.impact_decay_factor
    
    # 调整公式
    adjusted = base_rate + impact_factor * (1 - base_rate)
    
    # 限制范围
    min_rate, max_rate = config.decay_rate_range
    return max(min_rate, min(max_rate, adjusted))


def apply_decay(
    current_strength: float,
    decay_rate: float,
    time_days: float
) -> float:
    """
    应用衰减计算
    
    公式: strength_after = strength_before * decay_rate^time_days
    
    Args:
        current_strength: 当前强度
        decay_rate: 衰减率
        time_days: 经过的天数
    
    Returns:
        衰减后的强度
    """
    return current_strength * (decay_rate ** time_days)


def calculate_multi_event_decay(
    neurons: List[Dict],
    reference_time: Optional[datetime] = None
) -> List[Tuple[uuid.UUID, float]]:
    """
    计算多个神经元的衰减
    
    Args:
        neurons: 神经元列表，每项包含 id, event_type, impact_score, current_strength
        reference_time: 参考时间（用于批量计算）
    
    Returns:
        [(neuron_id, new_strength), ...]
    """
    if reference_time is None:
        reference_time = datetime.now()
    
    results = []
    for neuron in neurons:
        decay_rate = calculate_decay_rate(
            neuron['event_type'],
            neuron.get('impact_score', 0.5)
        )
        
        # 计算时间差
        if 'last_decay_at' in neuron and neuron['last_decay_at']:
            last_time = neuron['last_decay_at']
            if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time)
            time_days = (reference_time - last_time).total_seconds() / (24 * 3600)
        else:
            time_days = 0.0
        
        new_strength = apply_decay(
            neuron['current_strength'],
            decay_rate,
            time_days
        )
        
        results.append((neuron['id'], new_strength))
    
    return results


class DecayScheduler:
    """
    衰减调度器
    
    负责:
    1. 管理所有神经元的衰减状态
    2. 批量应用衰减
    3. 生成衰减报告
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._decay_states: Dict[uuid.UUID, NeuronDecayState] = {}
    
    def register_neuron(
        self,
        event_id: uuid.UUID,
        event_type: str,
        impact_score: float = 0.5,
        created_at: Optional[datetime] = None
    ) -> NeuronDecayState:
        """
        注册新神经元，设置初始衰减状态
        
        Args:
            event_id: 神经元事件 ID
            event_type: 事件类型
            impact_score: 冲击力评分
            created_at: 创建时间
        
        Returns:
            创建的衰减状态
        """
        if event_id in self._decay_states:
            return self._decay_states[event_id]
        
        base_rate = self.config.base_decay_rates.get(event_type, 0.995)
        now = created_at or datetime.now()
        
        state = NeuronDecayState(
            event_id=event_id,
            event_type=event_type,
            base_decay_rate=base_rate,
            impact_score=impact_score,
            decay_rate=calculate_decay_rate(event_type, impact_score, self.config),
            created_at=now,
            last_decay_at=now
        )
        
        self._decay_states[event_id] = state
        return state
    
    def get_decay_state(self, event_id: uuid.UUID) -> Optional[NeuronDecayState]:
        """获取神经元的衰减状态"""
        return self._decay_states.get(event_id)
    
    def update_impact_score(self, event_id: uuid.UUID, new_impact: float) -> None:
        """
        更新神经元的冲击力评分
        
        冲击力可以被外部事件增强（如强烈的情绪反应）
        """
        state = self._decay_states.get(event_id)
        if state:
            state.impact_score = max(0.0, min(1.0, new_impact))
            state.decay_rate = state.calculate_adjusted_decay_rate(self.config)
    
    def apply_decay_to_neuron(
        self,
        event_id: uuid.UUID,
        current_strength: float,
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        对单个神经元应用衰减
        
        Args:
            event_id: 神经元 ID
            current_strength: 当前强度
            reference_time: 参考时间
        
        Returns:
            衰减后的强度
        """
        state = self._decay_states.get(event_id)
        if state is None:
            # 未注册的神经元，使用默认衰减率
            return current_strength * 0.995
        
        time_days = state.time_elapsed_days(reference_time)
        new_strength = apply_decay(current_strength, state.decay_rate, time_days)
        
        # 更新上次衰减时间
        if reference_time:
            state.last_decay_at = reference_time
        else:
            state.last_decay_at = datetime.now()
        
        return new_strength
    
    def apply_decay_batch(
        self,
        neurons: List[Tuple[uuid.UUID, float]],
        reference_time: Optional[datetime] = None
    ) -> Dict[uuid.UUID, float]:
        """
        批量应用衰减
        
        Args:
            neurons: [(event_id, current_strength), ...]
            reference_time: 参考时间
        
        Returns:
            {event_id: new_strength, ...}
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        results = {}
        for event_id, strength in neurons:
            results[event_id] = self.apply_decay_to_neuron(
                event_id, strength, reference_time
            )
        
        return results
    
    def get_decay_ranking(self, top_k: Optional[int] = None) -> List[Tuple[uuid.UUID, float]]:
        """
        获取衰减最慢（最稳定）的神经元
        
        按 decay_rate 降序排列
        """
        rankings = [
            (event_id, state.decay_rate)
            for event_id, state in self._decay_states.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            return rankings[:top_k]
        return rankings
    
    def get_decay_statistics(self) -> Dict:
        """获取衰减系统统计信息"""
        if not self._decay_states:
            return {
                'total_neurons': 0,
                'avg_decay_rate': 0,
            }
        
        rates = [s.decay_rate for s in self._decay_states.values()]
        impacts = [s.impact_score for s in self._decay_states.values()]
        
        return {
            'total_neurons': len(self._decay_states),
            'avg_decay_rate': sum(rates) / len(rates),
            'min_decay_rate': min(rates),
            'max_decay_rate': max(rates),
            'avg_impact_score': sum(impacts) / len(impacts),
            'by_event_type': {
                et: sum(1 for s in self._decay_states.values() if s.event_type == et)
                for et in set(s.event_type for s in self._decay_states.values())
            }
        }


# ============== 衰减曲线示例 ==============

def estimate_decay_curve(
    event_type: str,
    impact_score: float,
    initial_strength: float = 1.0,
    days: int = 30
) -> List[Tuple[int, float]]:
    """
    生成衰减曲线（用于分析和可视化）
    
    Args:
        event_type: 事件类型
        impact_score: 冲击力评分
        initial_strength: 初始强度
        days: 计算天数
    
    Returns:
        [(day, strength), ...]
    """
    decay_rate = calculate_decay_rate(event_type, impact_score)
    curve = []
    
    for day in range(days + 1):
        strength = apply_decay(initial_strength, decay_rate, day)
        curve.append((day, strength))
    
    return curve


# ============== 场景模拟 ==============

def simulate_near_miss_decay() -> None:
    """
    模拟"差点被撞"场景
    - 高冲击力（0.9）
    - experience 类型
    - 期望：极慢衰减
    """
    decay_rate = calculate_decay_rate('experience', 0.9)
    print(f"[差点被撞] 衰减率: {decay_rate:.6f} (期望 > 0.999)")
    
    # 30 天后的强度
    strength_30d = apply_decay(1.0, decay_rate, 30)
    print(f"[差点被撞] 30天后强度: {strength_30d:.6f}")
    
    return decay_rate


def simulate_daily_walk_decay() -> None:
    """
    模拟"每天遛狗"场景
    - 低冲击力（0.3）
    - experience 类型
    - 期望：较快衰减
    """
    decay_rate = calculate_decay_rate('experience', 0.3)
    print(f"[每天遛狗] 衰减率: {decay_rate:.6f} (期望 < 0.99)")
    
    # 30 天后的强度
    strength_30d = apply_decay(1.0, decay_rate, 30)
    print(f"[每天遛狗] 30天后强度: {strength_30d:.6f}")
    
    return decay_rate


# ============== 全局实例 ==============

_default_scheduler: Optional[DecayScheduler] = None


def get_global_scheduler() -> DecayScheduler:
    """获取全局衰减调度器实例"""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = DecayScheduler()
    return _default_scheduler


def reset_global_scheduler() -> None:
    """重置全局衰减调度器"""
    global _default_scheduler
    if _default_scheduler:
        DecayScheduler.__init__(_default_scheduler)
