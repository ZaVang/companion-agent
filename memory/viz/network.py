"""
记忆网络可视化

提供网络结构可视化和统计分析。
支持 graphviz 和 networkx 两种后端。
"""

from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum

from memory.utils import now as utc_now


class VisualizerBackend(str, Enum):
    """可视化后端"""
    GRAPHVIZ = "graphviz"
    NETWORKX = "networkx"
    TEXT = "text"  # 纯文本输出


class NetworkStats(BaseModel):
    """网络统计信息"""
    total_neurons: int = 0
    total_connections: int = 0
    neurons_by_type: Dict[str, int] = Field(default_factory=dict)
    avg_connections_per_neuron: float = 0.0
    max_connections: int = 0
    isolated_neurons: int = 0
    density: float = 0.0  # 网络密度
    
    # 活跃度统计
    active_neurons: int = 0
    inactive_neurons: int = 0
    avg_strength: float = 0.0
    
    # 拓扑统计
    strongly_connected_components: int = 0
    average_clustering_coefficient: float = 0.0


@dataclass
class NeuronNode:
    """神经元节点"""
    id: str
    label: str
    neuron_type: str
    strength: float
    elo: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.neuron_type,
            'strength': self.strength,
            'elo': self.elo
        }


@dataclass
class ConnectionEdge:
    """连接边"""
    source: str
    target: str
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight
        }


class NetworkVisualizer:
    """记忆网络可视化器"""
    
    def __init__(
        self,
        backend: VisualizerBackend = VisualizerBackend.TEXT,
        max_nodes: int = 100
    ):
        self.backend = backend
        self.max_nodes = max_nodes
        
        # 缓存
        self._nodes: List[NeuronNode] = []
        self._edges: List[ConnectionEdge] = []
    
    def add_neuron(
        self,
        neuron_id: str,
        label: str,
        neuron_type: str,
        strength: float,
        elo: Optional[float] = None
    ) -> None:
        """添加神经元节点"""
        if len(self._nodes) < self.max_nodes:
            node = NeuronNode(
                id=neuron_id,
                label=label[:50],  # 截断长标签
                neuron_type=neuron_type,
                strength=strength,
                elo=elo
            )
            self._nodes.append(node)
    
    def add_connection(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0
    ) -> None:
        """添加连接边"""
        # 检查节点是否存在
        node_ids = {n.id for n in self._nodes}
        if source_id in node_ids or target_id in node_ids:
            edge = ConnectionEdge(
                source=source_id,
                target=target_id,
                weight=weight
            )
            self._edges.append(edge)
    
    def build_from_engram(
        self,
        engram,
        include_types: List[str] = None
    ) -> 'NetworkVisualizer':
        """
        从 Engram 构建可视化网络
        
        Args:
            engram: Engram 对象
            include_types: 要包含的事件类型列表
        """
        if include_types is None:
            include_types = ['chat', 'thought', 'reflection', 'perception', 'experience']
        
        self._nodes = []
        self._edges = []
        
        for event_type in include_types:
            neurons = engram.engram.get(event_type, [])
            for neuron in neurons:
                # 创建节点标签
                label = f"{event_type}:{neuron.event_id}"
                
                self.add_neuron(
                    neuron_id=str(neuron.event_id),
                    label=label,
                    neuron_type=event_type,
                    strength=neuron.strength,
                    elo=getattr(neuron, 'elo', None)
                )
                
                # 添加连接
                for conn in neuron.outgoing_connections:
                    self.add_connection(
                        source_id=str(neuron.event_id),
                        target_id=str(conn.target_id),
                        weight=neuron.strength
                    )
        
        return self
    
    def build_from_neurons(
        self,
        neurons: List,
        connections: List[Tuple[str, str]] = None
    ) -> 'NetworkVisualizer':
        """从神经元列表构建可视化网络"""
        self._nodes = []
        self._edges = []
        
        for neuron in neurons:
            # 支持 NeuronCell 对象和字典
            if hasattr(neuron, 'event_id'):
                neuron_id = str(neuron.event_id)
                event_type = getattr(neuron, 'event_type', 'unknown')
                strength = getattr(neuron, 'strength', 0.5)
                elo = getattr(neuron, 'elo', None)
            elif isinstance(neuron, dict):
                neuron_id = str(neuron.get('id', ''))
                event_type = neuron.get('type', neuron.get('event_type', 'unknown'))
                strength = neuron.get('strength', 0.5)
                elo = neuron.get('elo', None)
            else:
                continue
            
            label = f"{event_type}:{neuron_id[:8]}"
            
            self.add_neuron(
                neuron_id=neuron_id,
                label=label,
                neuron_type=event_type,
                strength=strength,
                elo=elo
            )
        
        if connections:
            for source, target in connections:
                self.add_connection(source, target)
        
        return self
    
    def render_text(self) -> str:
        """渲染为纯文本格式"""
        lines = []
        lines.append("=" * 60)
        lines.append("Memory Network Visualization (Text Mode)")
        lines.append("=" * 60)
        lines.append(f"Total Nodes: {len(self._nodes)}")
        lines.append(f"Total Edges: {len(self._edges)}")
        lines.append("")
        
        # 节点统计
        type_counts: Dict[str, int] = {}
        for node in self._nodes:
            type_counts[node.neuron_type] = type_counts.get(node.neuron_type, 0) + 1
        
        lines.append("Nodes by Type:")
        for ntype, count in sorted(type_counts.items()):
            lines.append(f"  {ntype}: {count}")
        lines.append("")
        
        # 节点列表
        lines.append("Nodes:")
        for node in self._nodes[:20]:  # 最多显示 20 个
            lines.append(f"  [{node.neuron_type}] {node.id[:16]} (strength={node.strength:.2f})")
        
        if len(self._nodes) > 20:
            lines.append(f"  ... and {len(self._nodes) - 20} more")
        
        lines.append("")
        lines.append("Connections:")
        for edge in self._edges[:30]:  # 最多显示 30 个
            lines.append(f"  {edge.source[:8]} -> {edge.target[:8]} (w={edge.weight:.2f})")
        
        if len(self._edges) > 30:
            lines.append(f"  ... and {len(self._edges) - 30} more")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def render_graphviz(self, filename: str = None) -> str:
        """
        渲染为 Graphviz DOT 格式
        
        Args:
            filename: 可选，保存到文件
        
        Returns:
            DOT 格式字符串
        """
        lines = []
        lines.append("digraph MemoryNetwork {")
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, fontsize=10];")
        
        # 定义节点颜色
        color_map = {
            'chat': '#4ECDC4',
            'thought': '#45B7D1',
            'reflection': '#96CEB4',
            'perception': '#FFEAA7',
            'experience': '#DDA0DD',
            'unknown': '#CCCCCC'
        }
        
        # 节点
        for node in self._nodes:
            color = color_map.get(node.neuron_type, '#CCCCCC')
            size = max(0.5, min(2, node.strength))  # 缩放强度到 [0.5, 2]
            lines.append(
                f'  "{node.id}" [label="{node.label}", '
                f'fillcolor="{color}", style=filled, '
                f'fontsize={8 + int(size * 2)}];'
            )
        
        # 边
        for edge in self._edges:
            width = max(0.5, edge.weight)
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[penwidth={width:.1f}];'
            )
        
        lines.append("}")
        
        dot_content = "\n".join(lines)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(dot_content)
        
        return dot_content
    
    def get_statistics(self) -> NetworkStats:
        """获取网络统计信息"""
        if not self._nodes:
            return NetworkStats()
        
        # 按类型统计节点
        neurons_by_type: Dict[str, int] = {}
        total_connections = len(self._edges)
        total_connections_per_neuron = 0
        max_connections = 0
        isolated = 0
        strength_sum = 0.0
        
        connected_nodes: Set[str] = set()
        for edge in self._edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        for node in self._nodes:
            neurons_by_type[node.neuron_type] = neurons_by_type.get(node.neuron_type, 0) + 1
            strength_sum += node.strength
            
            # 统计该节点的连接数
            node_connections = sum(
                1 for e in self._edges
                if e.source == node.id or e.target == node.id
            )
            total_connections_per_neuron += node_connections
            max_connections = max(max_connections, node_connections)
            
            if node.id not in connected_nodes:
                isolated += 1
        
        avg_connections = total_connections_per_neuron / len(self._nodes) if self._nodes else 0
        
        # 计算网络密度
        n = len(self._nodes)
        max_possible_edges = n * (n - 1)  # 有向图
        density = total_connections / max_possible_edges if max_possible_edges > 0 else 0
        
        # 活跃度统计
        active_threshold = 0.3
        active_count = sum(1 for n in self._nodes if n.strength >= active_threshold)
        
        return NetworkStats(
            total_neurons=len(self._nodes),
            total_connections=total_connections,
            neurons_by_type=neurons_by_type,
            avg_connections_per_neuron=avg_connections,
            max_connections=max_connections,
            isolated_neurons=isolated,
            density=density,
            active_neurons=active_count,
            inactive_neurons=n - active_count,
            avg_strength=strength_sum / n if n > 0 else 0
        )


def visualize_engram_network(
    engram,
    backend: VisualizerBackend = VisualizerBackend.TEXT,
    filename: str = None
) -> str:
    """
    便捷函数：可视化 Engram 网络
    
    Args:
        engram: Engram 对象
        backend: 可视化后端
        filename: 保存文件路径（仅 graphviz 后端）
    
    Returns:
        可视化结果字符串
    """
    viz = NetworkVisualizer(backend=backend)
    viz.build_from_engram(engram)
    
    if backend == VisualizerBackend.GRAPHVIZ:
        return viz.render_graphviz(filename)
    else:
        return viz.render_text()


def get_network_statistics(engram) -> NetworkStats:
    """
    便捷函数：获取 Engram 网络统计
    
    Args:
        engram: Engram 对象
    
    Returns:
        NetworkStats 对象
    """
    viz = NetworkVisualizer()
    viz.build_from_engram(engram)
    return viz.get_statistics()
