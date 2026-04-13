"""
因果图核心

管理因果图的数据结构。
"""

from typing import Dict, List, Tuple, Set, Optional
from pydantic import BaseModel, Field, UUID1
from datetime import datetime
from collections import defaultdict

from memory.causal.config import CausalConfig


class CausalEdge(BaseModel):
    """因果边"""
    cause_id: str
    effect_id: str
    strength: float = 1.0           # 因果强度
    confidence: float = 1.0        # 置信度
    evidence_count: int = 1        # 证据数量
    temporal_delay_ms: int = 0     # 时间延迟（毫秒）
    created_at: datetime = Field(default_factory=datetime.now)
    
    def update(self, new_strength: float, evidence_count: int = 1) -> None:
        """更新因果强度"""
        self.strength = (self.strength * self.evidence_count + new_strength) / (self.evidence_count + 1)
        self.evidence_count += evidence_count
        # 更新置信度
        self.confidence = min(1.0, self.evidence_count / 10.0)


class CausalGraph(BaseModel):
    """
    因果图
    
    维护神经元之间的因果关系。
    """
    # 节点
    nodes: Set[str] = Field(default_factory=set)
    
    # 有向边: (cause_id, effect_id) -> CausalEdge
    edges: Dict[Tuple[str, str], CausalEdge] = Field(default_factory=dict)
    
    # 统计信息
    activation_counts: Dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    transition_counts: Dict[Tuple[str, str], int] = Field(default_factory=lambda: defaultdict(int))
    
    def add_node(self, node_id: str) -> None:
        """添加节点"""
        self.nodes.add(node_id)
        if node_id not in self.activation_counts:
            self.activation_counts[node_id] = 0
    
    def add_causal_link(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        temporal_delay_ms: int = 0
    ) -> None:
        """添加因果链接"""
        self.add_node(cause)
        self.add_node(effect)
        
        key = (cause, effect)
        if key in self.edges:
            self.edges[key].update(strength)
        else:
            self.edges[key] = CausalEdge(
                cause_id=cause,
                effect_id=effect,
                strength=strength,
                temporal_delay_ms=temporal_delay_ms
            )
        
        self.transition_counts[key] += 1
    
    def get_causal_parents(self, node: str) -> List[Tuple[str, float]]:
        """获取节点的因（父节点）"""
        parents = []
        for (cause, effect), edge in self.edges.items():
            if effect == node:
                parents.append((cause, edge.strength))
        parents.sort(key=lambda x: x[1], reverse=True)
        return parents
    
    def get_causal_children(self, node: str) -> List[Tuple[str, float]]:
        """获取节点的果（子节点）"""
        children = []
        for (cause, effect), edge in self.edges.items():
            if cause == node:
                children.append((effect, edge.strength))
        children.sort(key=lambda x: x[1], reverse=True)
        return children
    
    def get_edge(self, cause: str, effect: str) -> Optional[CausalEdge]:
        """获取因果边"""
        return self.edges.get((cause, effect))
    
    def has_edge(self, cause: str, effect: str) -> bool:
        """检查是否存在因果边"""
        return (cause, effect) in self.edges
    
    def infer_probable_effects(
        self,
        cause: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        推断可能的果
        
        返回: [(effect_id, strength, confidence), ...]
        """
        children = self.get_causal_children(cause)
        return [(c, s, self.edges[(cause, c)].confidence) for c, s in children[:top_k]]
    
    def infer_probable_causes(
        self,
        effect: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        推断可能的因
        
        返回: [(cause_id, strength, confidence), ...]
        """
        parents = self.get_causal_parents(effect)
        return [(p, s, self.edges[(p, effect)].confidence) for p, s in parents[:top_k]]
    
    def find_causal_chains(
        self,
        start: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        查找从起点出发的所有因果链
        """
        chains = []
        visited = set()
        
        def dfs(current: str, path: List[str], depth: int):
            if depth >= max_length:
                return
            children = self.get_causal_children(current)
            if not children:
                if len(path) > 1:
                    chains.append(path.copy())
                return
            
            for child, _ in children:
                if child not in visited:
                    visited.add(child)
                    path.append(child)
                    dfs(child, path, depth + 1)
                    path.pop()
                    visited.remove(child)
        
        visited.add(start)
        dfs(start, [start], 0)
        
        # 也包括单节点链
        if not chains:
            chains.append([start])
        
        return chains
    
    def get_causal_path(
        self,
        cause: str,
        effect: str,
        max_length: int = 5
    ) -> Optional[List[str]]:
        """
        查找从因到果的因果路径
        
        返回路径或 None（如果不存在）
        """
        if cause == effect:
            return [cause]
        
        visited = set()
        
        def dfs(current: str, path: List[str]) -> Optional[List[str]]:
            if len(path) > max_length:
                return None
            if current == effect:
                return path.copy()
            
            visited.add(current)
            for child, _ in self.get_causal_children(current):
                if child not in visited:
                    result = dfs(child, path + [child])
                    if result:
                        return result
            visited.remove(current)
            return None
        
        return dfs(cause, [cause])
    
    def to_adjacency_list(self) -> Dict[str, List[str]]:
        """转换为邻接表"""
        adj = defaultdict(list)
        for (cause, effect) in self.edges.keys():
            adj[cause].append(effect)
        return dict(adj)
    
    def get_statistics(self) -> dict:
        """获取图的统计信息"""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "avg_degree": sum(len(self.get_causal_children(n)) for n in self.nodes) / len(self.nodes) if self.nodes else 0,
            "total_transitions": sum(self.transition_counts.values()),
            "avg_confidence": sum(e.confidence for e in self.edges.values()) / len(self.edges) if self.edges else 0
        }
    
    class Config:
        arbitrary_types_allowed = True
