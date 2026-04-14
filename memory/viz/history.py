"""
记忆历史追踪

追踪记忆的变化历史，支持：
1. 神经元激活历史
2. 强度变化历史
3. Elo 变化历史
4. 连接变化历史
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json

from memory.utils import now as utc_now


class ChangeType(str, Enum):
    """变化类型"""
    CREATED = "created"
    ACTIVATED = "activated"
    DECAYED = "decayed"
    CONSOLIDATED = "consolidated"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ELO_CHANGED = "elo_changed"
    STRENGTH_CHANGED = "strength_changed"
    DELETED = "deleted"
    SPLIT = "split"
    ABSTRACTED = "abstracted"


class HistoryEntry(BaseModel):
    """历史记录条目"""
    timestamp: datetime = Field(default_factory=utc_now)
    neuron_id: str
    change_type: ChangeType
    
    # 变化详情
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    
    # 元数据
    event_type: Optional[str] = None  # 事件类型
    scene: Optional[str] = None       # 场景
    emotion: Optional[Dict] = None    # 情绪
    metadata: Dict = Field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'neuron_id': self.neuron_id,
            'change_type': self.change_type.value,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'event_type': self.event_type,
            'scene': self.scene,
            'emotion': self.emotion,
            'metadata': self.metadata
        }


class MemoryHistory:
    """记忆历史追踪器"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: List[HistoryEntry] = []
        
        # 索引
        self._by_neuron: Dict[str, List[int]] = defaultdict(list)  # neuron_id -> entry indices
        self._by_type: Dict[ChangeType, List[int]] = defaultdict(list)  # change_type -> indices
        self._by_time: Dict[str, List[int]] = defaultdict(list)  # date string -> indices
    
    def add_entry(
        self,
        neuron_id: str,
        change_type: ChangeType,
        old_value: Any = None,
        new_value: Any = None,
        event_type: str = None,
        scene: str = None,
        emotion: Dict = None,
        **metadata
    ) -> HistoryEntry:
        """添加历史记录"""
        entry = HistoryEntry(
            neuron_id=neuron_id,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            event_type=event_type,
            scene=scene,
            emotion=emotion,
            metadata=metadata
        )
        
        # 添加到列表
        idx = len(self._entries)
        self._entries.append(entry)
        
        # 更新索引
        self._by_neuron[neuron_id].append(idx)
        self._by_type[change_type].append(idx)
        self._by_time[entry.timestamp.strftime('%Y-%m-%d')].append(idx)
        
        # 清理旧记录
        if len(self._entries) > self.max_entries:
            self._cleanup_old_entries()
        
        return entry
    
    def _cleanup_old_entries(self) -> None:
        """清理超出限制的旧记录"""
        # 保留最近的 max_entries 条
        keep_count = self.max_entries
        self._entries = self._entries[-keep_count:]
        
        # 重建索引
        self._by_neuron.clear()
        self._by_type.clear()
        self._by_time.clear()
        
        for idx, entry in enumerate(self._entries):
            self._by_neuron[entry.neuron_id].append(idx)
            self._by_type[entry.change_type].append(idx)
            self._by_time[entry.timestamp.strftime('%Y-%m-%d')].append(idx)
    
    def get_neuron_history(
        self,
        neuron_id: str,
        limit: int = None
    ) -> List[HistoryEntry]:
        """获取指定神经元的所有历史记录"""
        indices = self._by_neuron.get(neuron_id, [])
        if limit:
            indices = indices[-limit:]
        return [self._entries[i] for i in indices]
    
    def get_by_change_type(
        self,
        change_type: ChangeType,
        limit: int = None
    ) -> List[HistoryEntry]:
        """获取指定变化类型的所有记录"""
        indices = self._by_type.get(change_type, [])
        if limit:
            indices = indices[-limit:]
        return [self._entries[i] for i in indices]
    
    def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime = None,
        neuron_id: str = None
    ) -> List[HistoryEntry]:
        """获取时间范围内的记录"""
        if end_time is None:
            end_time = utc_now()
        
        results = []
        for entry in self._entries:
            if start_time <= entry.timestamp <= end_time:
                if neuron_id is None or entry.neuron_id == neuron_id:
                    results.append(entry)
        
        return results
    
    def get_recent(self, count: int = 10) -> List[HistoryEntry]:
        """获取最近 N 条记录"""
        return self._entries[-count:] if self._entries else []
    
    def get_statistics(self) -> Dict:
        """获取历史统计"""
        total = len(self._entries)
        
        by_type: Dict[str, int] = {}
        for change_type, indices in self._by_type.items():
            # 安全处理：change_type 可能是枚举或字符串
            key = change_type.value if hasattr(change_type, 'value') else str(change_type)
            by_type[key] = len(indices)
        
        # 唯一神经元数
        unique_neurons = len(self._by_neuron)
        
        # 时间范围
        if self._entries:
            first = self._entries[0].timestamp
            last = self._entries[-1].timestamp
            time_range = (last - first).total_seconds() / 3600  # 小时
        else:
            first = last = None
            time_range = 0
        
        return {
            'total_entries': total,
            'by_change_type': by_type,
            'unique_neurons': unique_neurons,
            'first_entry': first.isoformat() if first else None,
            'last_entry': last.isoformat() if last else None,
            'time_range_hours': time_range
        }
    
    def export_json(self, filepath: str = None) -> str:
        """导出为 JSON 格式"""
        data = [entry.to_dict() for entry in self._entries]
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def clear(self) -> None:
        """清空历史"""
        self._entries.clear()
        self._by_neuron.clear()
        self._by_type.clear()
        self._by_time.clear()


class MemoryTracer:
    """内存追踪器（用于函数调用）"""
    
    def __init__(self, history: Optional[MemoryHistory] = None):
        self.history = history or MemoryHistory()
    
    def trace_activation(
        self,
        neuron_id: str,
        old_strength: float,
        new_strength: float,
        event_type: str = None,
        **metadata
    ) -> HistoryEntry:
        """追踪激活"""
        return self.history.add_entry(
            neuron_id=neuron_id,
            change_type=ChangeType.ACTIVATED,
            old_value=old_strength,
            new_value=new_strength,
            event_type=event_type,
            **metadata
        )
    
    def trace_decay(
        self,
        neuron_id: str,
        old_strength: float,
        new_strength: float,
        decay_rate: float,
        **metadata
    ) -> HistoryEntry:
        """追踪衰减"""
        return self.history.add_entry(
            neuron_id=neuron_id,
            change_type=ChangeType.DECAYED,
            old_value=old_strength,
            new_value=new_strength,
            metadata={'decay_rate': decay_rate, **metadata}
        )
    
    def trace_elo_change(
        self,
        neuron_id: str,
        old_elo: float,
        new_elo: float,
        reason: str = None,
        **metadata
    ) -> HistoryEntry:
        """追踪 Elo 变化"""
        return self.history.add_entry(
            neuron_id=neuron_id,
            change_type=ChangeType.ELO_CHANGED,
            old_value=old_elo,
            new_value=new_elo,
            metadata={'reason': reason, **metadata}
        )
    
    def trace_connection(
        self,
        source_id: str,
        target_id: str,
        connected: bool,
        **metadata
    ) -> HistoryEntry:
        """追踪连接变化"""
        return self.history.add_entry(
            neuron_id=source_id,
            change_type=ChangeType.CONNECTED if connected else ChangeType.DISCONNECTED,
            old_value=target_id if not connected else None,
            new_value=target_id if connected else None,
            **metadata
        )
    
    def trace_strength_change(
        self,
        neuron_id: str,
        old_strength: float,
        new_strength: float,
        reason: str = None,
        **metadata
    ) -> HistoryEntry:
        """追踪强度变化"""
        return self.history.add_entry(
            neuron_id=neuron_id,
            change_type=ChangeType.STRENGTH_CHANGED,
            old_value=old_strength,
            new_value=new_strength,
            metadata={'reason': reason, **metadata}
        )
    
    def create_decorator(
        self,
        neuron_id_extractor: Callable = None
    ):
        """
        创建追踪装饰器
        
        Args:
            neuron_id_extractor: 从函数参数提取 neuron_id 的函数
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # 尝试提取 neuron_id
                neuron_id = None
                if neuron_id_extractor:
                    neuron_id = neuron_id_extractor(*args, **kwargs)
                elif 'neuron_id' in kwargs:
                    neuron_id = kwargs['neuron_id']
                elif len(args) > 0:
                    neuron_id = str(args[0]) if args else None
                
                if neuron_id:
                    self.history.add_entry(
                        neuron_id=neuron_id,
                        change_type=ChangeType.ACTIVATED,
                        metadata={'function': func.__name__}
                    )
                
                return result
            return wrapper
        return decorator


# 全局追踪器实例
_global_tracer: Optional[MemoryTracer] = None


def get_global_tracer() -> MemoryTracer:
    """获取全局追踪器"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = MemoryTracer()
    return _global_tracer


def track_memory_change(
    neuron_id: str,
    change_type: ChangeType,
    old_value: Any = None,
    new_value: Any = None,
    **metadata
) -> HistoryEntry:
    """便捷函数：追踪记忆变化"""
    tracer = get_global_tracer()
    return tracer.history.add_entry(
        neuron_id=neuron_id,
        change_type=change_type,
        old_value=old_value,
        new_value=new_value,
        **metadata
    )
