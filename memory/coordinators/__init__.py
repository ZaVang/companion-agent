"""
MemorySystem 协调器模块

提供 Facade 模式下的职责分离，将 MemorySystem 的功能拆分到独立的协调器。
"""

from memory.coordinators.storage import StorageCoordinator
from memory.coordinators.retriever import RetrieverCoordinator
from memory.coordinators.lifecycle import LifecycleCoordinator
from memory.coordinators.batch import BatchOperations
from memory.coordinators.stats_viz import StatsVizCoordinator

__all__ = [
    "StorageCoordinator",
    "RetrieverCoordinator", 
    "LifecycleCoordinator",
    "BatchOperations",
    "StatsVizCoordinator",
]
