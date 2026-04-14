"""
神经元死亡机制

触发条件：
1. Elo 评分过低
2. 强度过低
3. 长期未激活
4. 与其他神经元失去连接（孤立）
"""

from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from memory.utils import now as utc_now, days_diff


class DeathReason(str, Enum):
    """死亡原因枚举"""
    LOW_ELO = "low_elo"                    # Elo 评分过低
    LOW_STRENGTH = "low_strength"         # 强度过低
    LONG_INACTIVE = "long_inactive"       # 长期未激活
    ISOLATED = "isolated"                 # 孤立神经元
    DECAY_BELOW_THRESHOLD = "decay_below" # 衰减到阈值以下


class DeathCriteria(BaseModel):
    """死亡判定标准"""
    min_elo: float = 100.0              # 最低 Elo 阈值
    min_strength: float = 0.05          # 最低强度阈值
    max_inactive_days: int = 90          # 最大未激活天数
    min_connections: int = 1            # 最小连接数（小于此值视为孤立）
    decay_death_threshold: float = 0.01  # 衰减致死阈值


class DeathRecord(BaseModel):
    """死亡记录"""
    neuron_id: str
    reason: DeathReason
    elo: Optional[float] = None
    strength: Optional[float] = None
    inactive_days: Optional[float] = None
    connections_count: int = 0
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: Dict = Field(default_factory=dict)


class NeuronDeath:
    """神经元死亡判定"""
    
    def __init__(self, criteria: Optional[DeathCriteria] = None):
        self.criteria = criteria or DeathCriteria()
    
    def evaluate(
        self,
        neuron_id: str,
        elo: Optional[float] = None,
        strength: Optional[float] = None,
        last_activation: Optional[datetime] = None,
        connections_count: int = 0,
        event_type: str = None
    ) -> Tuple[bool, Optional[DeathReason]]:
        """
        评估神经元是否应该死亡
        
        Returns:
            (should_die, reason)
        """
        # 1. 检查 Elo
        if elo is not None and elo < self.criteria.min_elo:
            return True, DeathReason.LOW_ELO
        
        # 2. 检查强度
        if strength is not None and strength < self.criteria.min_strength:
            return True, DeathReason.LOW_STRENGTH
        
        # 3. 检查活跃时间
        if last_activation is not None:
            inactive_days = days_diff(last_activation, utc_now())
            if inactive_days > self.criteria.max_inactive_days:
                return True, DeathReason.LONG_INACTIVE
        
        # 4. 检查是否孤立
        if connections_count < self.criteria.min_connections:
            return True, DeathReason.ISOLATED
        
        # 5. 检查衰减致死
        if strength is not None and strength < self.criteria.decay_death_threshold:
            return True, DeathReason.DECAY_BELOW_THRESHOLD
        
        return False, None
    
    def get_death_probability(
        self,
        elo: Optional[float] = None,
        strength: Optional[float] = None,
        inactive_days: Optional[float] = None,
        connections_count: int = 0
    ) -> float:
        """
        计算死亡概率（用于概率性死亡模型）
        
        返回 [0, 1] 之间的概率值
        """
        probabilities = []
        
        # Elo 因子
        if elo is not None:
            elo_prob = max(0, 1 - (elo - 100) / 1900)  # 归一化到 [0, 1]
            probabilities.append(elo_prob * 0.3)
        
        # 强度因子
        if strength is not None:
            strength_prob = max(0, 1 - strength * 10)  # 强度越低概率越高
            probabilities.append(strength_prob * 0.3)
        
        # 活跃时间因子
        if inactive_days is not None:
            if inactive_days > self.criteria.max_inactive_days:
                inactive_prob = min(1, (inactive_days - self.criteria.max_inactive_days) / 30)
                probabilities.append(inactive_prob * 0.3)
        
        # 连接因子
        if connections_count < self.criteria.min_connections:
            conn_prob = 1 - connections_count / self.criteria.min_connections
            probabilities.append(conn_prob * 0.1)
        
        return min(1, sum(probabilities))


def should_neuron_die(
    neuron_id: str,
    elo: Optional[float] = None,
    strength: Optional[float] = None,
    last_activation: Optional[datetime] = None,
    connections_count: int = 0,
    criteria: Optional[DeathCriteria] = None
) -> Tuple[bool, Optional[DeathReason]]:
    """
    便捷函数：判断神经元是否应该死亡
    """
    death_checker = NeuronDeath(criteria)
    return death_checker.evaluate(
        neuron_id=neuron_id,
        elo=elo,
        strength=strength,
        last_activation=last_activation,
        connections_count=connections_count
    )


class NeuronDeathManager:
    """神经元死亡管理器"""
    
    def __init__(self, criteria: Optional[DeathCriteria] = None):
        self.criteria = criteria or DeathCriteria()
        self.death_history: List[DeathRecord] = []
        self._death_checker = NeuronDeath(self.criteria)
    
    def register_death(self, record: DeathRecord) -> None:
        """记录神经元死亡"""
        self.death_history.append(record)
    
    def evaluate_neuron(
        self,
        neuron_id: str,
        elo: Optional[float] = None,
        strength: Optional[float] = None,
        last_activation: Optional[datetime] = None,
        connections_count: int = 0
    ) -> Tuple[bool, Optional[DeathReason], Optional[DeathRecord]]:
        """
        评估单个神经元
        
        Returns:
            (should_die, reason, record)
        """
        should_die, reason = self._death_checker.evaluate(
            neuron_id=neuron_id,
            elo=elo,
            strength=strength,
            last_activation=last_activation,
            connections_count=connections_count
        )
        
        if should_die and reason:
            record = DeathRecord(
                neuron_id=neuron_id,
                reason=reason,
                elo=elo,
                strength=strength,
                inactive_days=days_diff(last_activation, utc_now()) if last_activation else None,
                connections_count=connections_count
            )
            return True, reason, record
        
        return False, None, None
    
    def batch_evaluate(
        self,
        neurons: List[Dict]
    ) -> List[Tuple[str, bool, Optional[DeathReason], Optional[DeathRecord]]]:
        """
        批量评估神经元
        
        neuron 格式: {
            'id': str,
            'elo': float,  # 可选
            'strength': float,  # 可选
            'last_activation': datetime,  # 可选
            'connections_count': int
        }
        """
        results = []
        for neuron in neurons:
            should_die, reason, record = self.evaluate_neuron(
                neuron_id=neuron.get('id', ''),
                elo=neuron.get('elo'),
                strength=neuron.get('strength'),
                last_activation=neuron.get('last_activation'),
                connections_count=neuron.get('connections_count', 0)
            )
            results.append((neuron.get('id', ''), should_die, reason, record))
        
        return results
    
    def execute_deaths(
        self,
        neurons_to_die: List[Dict]
    ) -> List[DeathRecord]:
        """
        执行死亡操作
        
        Returns:
            死亡记录列表
        """
        records = []
        for neuron in neurons_to_die:
            should_die, reason, record = self.evaluate_neuron(
                neuron_id=neuron.get('id', ''),
                elo=neuron.get('elo'),
                strength=neuron.get('strength'),
                last_activation=neuron.get('last_activation'),
                connections_count=neuron.get('connections_count', 0)
            )
            
            if should_die:
                self.register_death(record)
                records.append(record)
        
        return records
    
    def get_statistics(self) -> Dict:
        """获取死亡统计"""
        if not self.death_history:
            return {
                'total_deaths': 0,
                'by_reason': {},
                'recent_deaths': 0
            }
        
        by_reason: Dict[str, int] = {}
        for record in self.death_history:
            reason = record.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        # 最近 7 天死亡数
        now = utc_now()
        recent_deaths = sum(
            1 for r in self.death_history
            if days_diff(r.timestamp, now) <= 7
        )
        
        return {
            'total_deaths': len(self.death_history),
            'by_reason': by_reason,
            'recent_deaths': recent_deaths
        }
