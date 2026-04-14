"""
神经元新生机制

触发条件：
1. 高强度记忆分裂
2. 概念抽象
3. 模式识别
"""

from typing import Dict, List, Optional, Tuple, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from memory.utils import now as utc_now


class BirthReason(str, Enum):
    """出生原因枚举"""
    MEMORY_SPLIT = "memory_split"          # 记忆分裂
    CONCEPT_ABSTRACTION = "concept_abstract"  # 概念抽象
    PATTERN_DETECTED = "pattern_detected"  # 模式识别
    REINFORCEMENT = "reinforcement"        # 强化生成
    CONTEXTUAL = "contextual"              # 上下文触发


class BirthCriteria(BaseModel):
    """新生判定标准"""
    min_split_strength: float = 0.8       # 分裂所需最小强度
    min_abstraction_count: int = 3        # 抽象所需最小关联数
    abstraction_similarity: float = 0.7   # 抽象所需相似度
    pattern_occurrence: int = 3           # 模式检测所需出现次数
    reinforcement_threshold: float = 0.9  # 强化生成阈值


class BirthRecord(BaseModel):
    """新生记录"""
    neuron_id: str
    reason: BirthReason
    parent_ids: List[str] = Field(default_factory=list)  # 父神经元 ID
    associated_neuron_ids: List[str] = Field(default_factory=list)  # 关联神经元
    content: str = ""
    metadata: Dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class NeuronBirth:
    """神经元新生判定"""
    
    def __init__(self, criteria: Optional[BirthCriteria] = None):
        self.criteria = criteria or BirthCriteria()
    
    def should_split(
        self,
        strength: float,
        connections_count: int = 0
    ) -> bool:
        """
        判断是否应该分裂
        
        高强度记忆可能分裂成多个更具体的记忆
        """
        return strength > self.criteria.min_split_strength
    
    def should_abstract(
        self,
        related_neurons: List[Dict],
        content: str = ""
    ) -> bool:
        """
        判断是否应该抽象
        
        多个相似记忆可以抽象成一个更通用的概念
        """
        if len(related_neurons) < self.criteria.min_abstraction_count:
            return False
        
        # 检查相似度（简化版，实际可使用 embedding 相似度）
        # 这里假设 content 已经提供了足够的相似度信息
        return True
    
    def should_create_pattern_neuron(
        self,
        occurrence_count: int,
        similar_content_count: int
    ) -> bool:
        """
        判断是否应该创建模式神经元
        
        重复出现的模式可以生成专门的模式神经元
        """
        return (
            occurrence_count >= self.criteria.pattern_occurrence and
            similar_content_count >= 2
        )
    
    def get_birth_probability(
        self,
        strength: float,
        related_count: int,
        occurrence_count: int = 1
    ) -> float:
        """
        计算新生概率
        """
        prob = 0.0
        
        # 强度因子
        if strength > self.criteria.min_split_strength:
            prob += 0.3 * (strength - self.criteria.min_split_strength)
        
        # 关联因子
        if related_count >= self.criteria.min_abstraction_count:
            prob += 0.3 * min(1, related_count / 5)
        
        # 出现次数因子
        if occurrence_count >= self.criteria.pattern_occurrence:
            prob += 0.4 * min(1, occurrence_count / 5)
        
        return min(1, prob)


def should_create_new_neuron(
    strength: float = 0,
    related_count: int = 0,
    occurrence_count: int = 1,
    criteria: Optional[BirthCriteria] = None
) -> bool:
    """
    便捷函数：判断是否应该创建新神经元
    """
    birth_checker = NeuronBirth(criteria)
    prob = birth_checker.get_birth_probability(
        strength=strength,
        related_count=related_count,
        occurrence_count=occurrence_count
    )
    return prob > 0.3  # 简单阈值


def split_high_intensity_memory(
    neuron_id: str,
    content: str,
    strength: float,
    criteria: Optional[BirthCriteria] = None
) -> List[Tuple[str, str]]:
    """
    分裂高强度记忆
    
    将一个高强度记忆分裂成多个更具体的子记忆
    
    Returns:
        List[(sub_content, new_neuron_id)]
    """
    if criteria is None:
        criteria = BirthCriteria()
    
    if strength < criteria.min_split_strength:
        return []
    
    # 简单实现：按句子或段落分裂
    # 实际实现可能需要使用 NLP 技术
    parts = []
    if len(content) > 100:
        # 按逗号、句号分割
        sentences = content.replace('。', '.|').replace('，', '.|').replace('\n', '.|').split('.|')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 2:
            parts = sentences[:3]  # 最多分裂成 3 个子记忆
        else:
            parts = [content[:len(content)//2], content[len(content)//2:]]
    else:
        parts = [content]
    
    # 生成新神经元 ID
    import uuid
    results = []
    for part in parts:
        if part:
            results.append((part, str(uuid.uuid4())))
    
    return results


def abstract_concept(
    related_contents: List[str],
    concept_type: str = "general"
) -> str:
    """
    从多个相关记忆抽象出一个概念
    
    Args:
        related_contents: 相关记忆内容列表
        concept_type: 概念类型 (general, temporal, spatial, causal)
    
    Returns:
        抽象后的概念描述
    """
    if not related_contents:
        return ""
    
    if len(related_contents) == 1:
        return related_contents[0]
    
    # 简单实现：取共同关键词
    # 实际实现可能需要使用 LLM 或 NLP 技术
    common_words = set()
    for content in related_contents:
        words = content.split()
        if not common_words:
            common_words = set(words[:5])  # 初始化
        else:
            common_words &= set(words[:5])
    
    if common_words:
        return f"{concept_type}: {' '.join(list(common_words)[:3])}"
    
    # 回退：简单拼接
    return f"{concept_type}: {'; '.join(related_contents[:2])}"


class NeuronBirthManager:
    """神经元新生管理器"""
    
    def __init__(self, criteria: Optional[BirthCriteria] = None):
        self.criteria = criteria or BirthCriteria()
        self.birth_history: List[BirthRecord] = []
        self._birth_checker = NeuronBirth(self.criteria)
        
        # 模式追踪
        self._pattern_tracker: Dict[str, int] = {}  # pattern_key -> count
        self._content_similarity_cache: Dict[str, List[str]] = {}  # content -> similar_contents
    
    def register_birth(self, record: BirthRecord) -> None:
        """记录神经元新生"""
        self.birth_history.append(record)
    
    def evaluate_birth(
        self,
        neuron_id: str,
        strength: float,
        related_neurons: List[Dict] = None,
        related_contents: List[str] = None,
        reason: BirthReason = BirthReason.MEMORY_SPLIT,
        content: str = ""
    ) -> Tuple[bool, Optional[BirthRecord]]:
        """
        评估是否应该生成新神经元
        """
        if related_neurons is None:
            related_neurons = []
        if related_contents is None:
            related_contents = []
        
        should_birth = False
        actual_reason = reason
        
        # 1. 检查是否应该分裂
        if self._birth_checker.should_split(strength, len(related_neurons)):
            should_birth = True
            actual_reason = BirthReason.MEMORY_SPLIT
        
        # 2. 检查是否应该抽象
        elif related_contents and self._birth_checker.should_abstract(related_neurons, content):
            should_birth = True
            actual_reason = BirthReason.CONCEPT_ABSTRACTION
        
        # 3. 检查模式
        pattern_key = content[:50] if content else neuron_id
        if pattern_key in self._pattern_tracker:
            self._pattern_tracker[pattern_key] += 1
        else:
            self._pattern_tracker[pattern_key] = 1
        
        if self._birth_checker.should_create_pattern_neuron(
            self._pattern_tracker[pattern_key],
            len(related_neurons)
        ):
            should_birth = True
            actual_reason = BirthReason.PATTERN_DETECTED
        
        if should_birth:
            record = BirthRecord(
                neuron_id=neuron_id,
                reason=actual_reason,
                parent_ids=[neuron_id],
                associated_neuron_ids=[n.get('id', '') for n in related_neurons],
                content=content,
                metadata={
                    'strength': strength,
                    'related_count': len(related_neurons)
                }
            )
            return True, record
        
        return False, None
    
    def track_pattern(self, pattern_key: str) -> int:
        """追踪模式出现次数"""
        self._pattern_tracker[pattern_key] = self._pattern_tracker.get(pattern_key, 0) + 1
        return self._pattern_tracker[pattern_key]
    
    def execute_birth(
        self,
        neuron_id: str,
        strength: float,
        content: str = "",
        related_contents: List[str] = None,
        reason: BirthReason = BirthReason.MEMORY_SPLIT
    ) -> Optional[BirthRecord]:
        """
        执行新生操作
        """
        if related_contents is None:
            related_contents = []
        
        should_birth, record = self.evaluate_birth(
            neuron_id=neuron_id,
            strength=strength,
            related_contents=related_contents,
            reason=reason,
            content=content
        )
        
        if should_birth and record:
            self.register_birth(record)
            return record
        
        return None
    
    def get_statistics(self) -> Dict:
        """获取新生统计"""
        if not self.birth_history:
            return {
                'total_births': 0,
                'by_reason': {},
                'recent_births': 0
            }
        
        from memory.utils import days_diff
        
        by_reason: Dict[str, int] = {}
        for record in self.birth_history:
            reason = record.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        now = utc_now()
        recent_births = sum(
            1 for r in self.birth_history
            if days_diff(r.timestamp, now) <= 7
        )
        
        return {
            'total_births': len(self.birth_history),
            'by_reason': by_reason,
            'recent_births': recent_births,
            'pattern_tracker_size': len(self._pattern_tracker)
        }
