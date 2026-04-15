"""
统计可视化协调器 - 管理统计信息和网络可视化

负责：
- 系统统计
- Graphviz 导出
- 网络统计
- 网络可视化
"""

from typing import TYPE_CHECKING, Dict, List, Any, Tuple

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.elo import EloCompetition
    from memory.dynamics import NeuronDynamics, NeuronDeathManager, NeuronBirthManager
    from memory.viz import MemoryHistory, NetworkVisualizer
    from memory.indexing import MemoryIndex
    from memory.coordinators.storage import StorageCoordinator


class StatsVizCoordinator:
    """
    管理 MemorySystem 的统计和可视化。
    
    职责：
    - 系统统计
    - Graphviz DOT 导出
    - 网络统计
    - 网络可视化
    """
    
    def __init__(
        self,
        storage: "StorageCoordinator",
        elo: "EloCompetition",
        dynamics: "NeuronDynamics",
        history: "MemoryHistory",
        visualizer: "NetworkVisualizer",
        index: Any = None,
    ) -> None:
        self.storage = storage
        self.elo = elo
        self.dynamics = dynamics
        self.history = history
        self.visualizer = visualizer
        self.index = index
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        storage_stats = self.storage.get_statistics()
        
        # Elo 统计
        elo_stats = {}
        if hasattr(self.elo, 'get_statistics'):
            elo_stats = self.elo.get_statistics()
        
        # Dynamics 统计
        dynamics_stats = {}
        if hasattr(self.dynamics, 'death_manager') and hasattr(self.dynamics.death_manager, 'get_statistics'):
            dynamics_stats = {
                'total_deaths': self.dynamics.death_manager.get_statistics().get('total_deaths', 0),
                'total_births': getattr(self.dynamics.birth_manager, 'get_statistics', lambda: {})().get('total_births', 0)
            }
        
        # History 统计
        history_stats = {}
        if self.history and hasattr(self.history, 'get_statistics'):
            history_stats = self.history.get_statistics()
        
        # Index 统计
        index_stats = None
        if self.index and hasattr(self.index, 'get_statistics'):
            index_stats = self.index.get_statistics()
        
        return {
            **storage_stats,
            'elo_statistics': elo_stats,
            'dynamics': dynamics_stats,
            'history': history_stats,
            'index': index_stats,
        }
    
    def to_graphviz(self) -> str:
        """
        导出记忆网络为 Graphviz DOT 格式字符串。
        
        Returns:
            DOT 格式字符串（包含 "digraph" 关键字）
        """
        neurons = list(self.storage.get_all_neurons().values())
        connections = [
            (str(n.event_id), str(c.target_id))
            for n in neurons
            for c in n.outgoing_connections
        ]
        self.visualizer.build_from_neurons(neurons, connections)
        return self.visualizer.render_graphviz()
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取记忆网络的统计信息。
        
        Returns:
            dict 包含 node_count, edge_count, avg_degree 等
        """
        neurons = list(self.storage.get_all_neurons().values())
        connections = [
            (str(n.event_id), str(c.target_id))
            for n in neurons
            for c in n.outgoing_connections
        ]
        self.visualizer.build_from_neurons(neurons, connections)
        stats = self.visualizer.get_statistics()
        
        return {
            'node_count': stats.total_neurons,
            'edge_count': stats.total_connections,
            'avg_degree': stats.avg_connections_per_neuron,
            'density': stats.density,
            'isolated_nodes': stats.isolated_neurons,
            'neurons_by_type': stats.neurons_by_type,
            'avg_strength': stats.avg_strength,
        }
    
    def get_network_visualization(self) -> str:
        """获取网络可视化文本"""
        neurons = list(self.storage.get_all_neurons().values())
        connections = []
        
        for neuron in neurons:
            for conn in neuron.outgoing_connections:
                connections.append((str(neuron.event_id), str(conn.target_id)))
        
        self.visualizer.build_from_neurons(neurons, connections)
        return self.visualizer.render_text()
