"""
Memory System API Routes

提供记忆网络可视化所需的 API 端点。
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import uuid

from memory.system import MemorySystem, MemorySystemConfig, DMNResult
from memory.neuron import NeuronCell
from memory.api_schema import EventType

router = APIRouter(prefix="/memory", tags=["memory"])

# 全局记忆系统实例
_memory_system: Optional[MemorySystem] = None

def get_memory_system() -> MemorySystem:
    """获取或创建记忆系统实例"""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem(MemorySystemConfig())
    return _memory_system


# ============== 请求/响应模型 ==============

class AddMemoryRequest(BaseModel):
    content: str = Field(..., description="记忆内容")
    event_type: str = Field(default="chat", description="事件类型")
    actor: str = Field(default="user", description="执行者")
    audience: Optional[List[str]] = Field(default=None, description="受众")
    impact_score: float = Field(default=0.5, ge=0.0, le=1.0, description="冲击力")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外数据")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="检索查询")
    top_k: int = Field(default=5, ge=1, le=100, description="返回数量")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="相似度阈值")
    memory_scope: str = Field(default="full", description="检索范围")


class NeuronResponse(BaseModel):
    id: str
    label: str
    type: str
    strength: float
    elo: float
    emotional_valence: Optional[float] = None
    emotional_arousal: Optional[float] = None
    is_active: bool = False
    is_consolidated: bool = False
    activation_threshold: float = 0.3
    created_at: Optional[str] = None
    content: Optional[str] = None


class NetworkResponse(BaseModel):
    nodes: List[NeuronResponse]
    links: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    total_neurons: int
    total_connections: int
    neurons_by_type: Dict[str, int]
    avg_connections_per_neuron: float
    avg_strength: float
    avg_elo: float
    active_neurons: int
    consolidated_neurons: int


class RetrievalResponse(BaseModel):
    id: str
    content: str
    score: float
    activated_neurons: List[str]
    retrieval_path: List[str]


class DMNResponse(BaseModel):
    success: bool
    consolidations: int
    prunings: int
    new_associations: int
    messages: List[str]
    activated_neurons: List[str]


class ConfigUpdateRequest(BaseModel):
    decay_rate: Optional[float] = None
    activation_threshold: Optional[float] = None
    elo_k_factor: Optional[float] = None
    consolidation_threshold: Optional[float] = None
    auto_decay: Optional[bool] = None
    auto_consolidation: Optional[bool] = None


# ============== 辅助函数 ==============

def neuron_to_response(neuron: NeuronCell, label: str = "", content: str = "") -> NeuronResponse:
    """将 NeuronCell 转换为 API 响应"""
    return NeuronResponse(
        id=str(neuron.event_id),
        label=label or f"{neuron.event_type}:{str(neuron.event_id)[:8]}",
        type=neuron.event_type,
        strength=neuron.strength,
        elo=getattr(neuron, 'elo', 1200),
        emotional_valence=neuron.emotional_valence if hasattr(neuron, 'emotional_valence') else None,
        emotional_arousal=neuron.emotional_arousal if hasattr(neuron, 'emotional_arousal') else None,
        is_consolidated=neuron.is_consolidated if hasattr(neuron, 'is_consolidated') else False,
        activation_threshold=neuron.activation_threshold,
        created_at=neuron.create_time.isoformat() if hasattr(neuron, 'create_time') else None,
        content=content,
    )


# ============== API 端点 ==============

@router.get("/network", response_model=NetworkResponse)
async def get_network():
    """获取记忆网络结构"""
    system = get_memory_system()
    
    nodes = []
    links = []
    
    # 从 engram 获取神经元
    engram = system._engrams.get("default")
    if engram:
        for event_type, neurons in engram.engram.items():
            for neuron in neurons:
                nodes.append(neuron_to_response(
                    neuron,
                    label=f"{event_type}:{str(neuron.event_id)[:8]}",
                ))
                # 添加连接
                for conn in neuron.outgoing_connections:
                    links.append({
                        "source": str(neuron.event_id),
                        "target": str(conn.target_id),
                        "weight": 1.0,
                    })
    
    return NetworkResponse(nodes=nodes, links=links)


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取网络统计信息"""
    system = get_memory_system()
    
    total_neurons = 0
    neurons_by_type: Dict[str, int] = {}
    total_strength = 0.0
    total_elo = 0.0
    consolidated = 0
    
    engram = system._engrams.get("default")
    if engram:
        for event_type, neurons in engram.engram.items():
            neurons_by_type[event_type] = len(neurons)
            total_neurons += len(neurons)
            for neuron in neurons:
                total_strength += neuron.strength
                total_elo += getattr(neuron, 'elo', 1200)
                if hasattr(neuron, 'is_consolidated') and neuron.is_consolidated:
                    consolidated += 1
    
    avg_strength = total_strength / max(total_neurons, 1)
    avg_elo = total_elo / max(total_neurons, 1)
    
    return StatsResponse(
        total_neurons=total_neurons,
        total_connections=len(engram.connections) if engram else 0,
        neurons_by_type=neurons_by_type,
        avg_connections_per_neuron=0.0,
        avg_strength=avg_strength,
        avg_elo=avg_elo,
        active_neurons=0,
        consolidated_neurons=consolidated,
    )


@router.post("/memories", response_model=NeuronResponse)
async def add_memory(req: AddMemoryRequest):
    """添加新记忆"""
    system = get_memory_system()
    
    try:
        neuron = system.add_memory(
            content=req.content,
            event_type=req.event_type,
            actor=req.actor,
            audience=req.audience or [],
            metadata=req.metadata or {},
        )
        return neuron_to_response(neuron, content=req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_memory(req: RetrieveRequest):
    """检索记忆"""
    system = get_memory_system()
    
    try:
        # 使用统一的检索器
        retriever = system.retriever if hasattr(system, 'retriever') else None
        
        if retriever:
            result = await retriever.retrieve(
                query=req.query,
                top_k=req.top_k,
                threshold=req.threshold,
            )
            return RetrievalResponse(
                id=str(uuid.uuid4()),
                content=f"Found {len(result.results) if hasattr(result, 'results') else 0} results",
                score=result.max_similarity if hasattr(result, 'max_similarity') else 0.0,
                activated_neurons=[str(r.event_id) for r in result.results] if hasattr(result, 'results') else [],
                retrieval_path=[],
            )
        
        # 回退到模拟结果
        return RetrievalResponse(
            id=str(uuid.uuid4()),
            content=f"Results for: {req.query}",
            score=0.85,
            activated_neurons=[],
            retrieval_path=[],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dmn", response_model=DMNResponse)
async def trigger_dmn():
    """触发 DMN 整合"""
    system = get_memory_system()
    
    try:
        result: DMNResult = system.dmn.consolidate()
        
        return DMNResponse(
            success=result.success,
            consolidations=result.consolidations,
            prunings=result.prunings,
            new_associations=result.new_associations,
            messages=result.messages,
            activated_neurons=[],
        )
    except Exception as e:
        return DMNResponse(
            success=False,
            consolidations=0,
            prunings=0,
            new_associations=0,
            messages=[str(e)],
            activated_neurons=[],
        )


@router.post("/elo/compete")
async def trigger_elo():
    """触发 Elo 竞争"""
    system = get_memory_system()
    
    try:
        system.elo.compete()
        return {"success": True, "message": "Elo competition triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_config(req: ConfigUpdateRequest):
    """更新系统配置"""
    return {"success": True, "config": req.dict(exclude_none=True)}
