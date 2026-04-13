"""
LongMemEval 兼容接口

定义与 LongMemEval 评测标准对齐的 API schema。

LongMemEval 五大核心能力:
1. Information Extraction (IE) - 从长对话中提取特定信息
2. Multi-Session Reasoning (MR) - 跨会话整合信息
3. Temporal Reasoning (TR) - 时间感知推理
4. Knowledge Updates (KU) - 动态更新知识
5. Abstention (ABS) - 识别未知信息
"""

from typing import Dict, List, Literal, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, UUID1
from enum import Enum


# ============== 枚举定义 ==============

class MemoryCapability(str, Enum):
    """记忆能力类型（对应 LongMemEval）"""
    INFORMATION_EXTRACTION = "information_extraction"      # 信息提取
    MULTI_SESSION_REASONING = "multi_session_reasoning"   # 跨会话推理
    TEMPORAL_REASONING = "temporal_reasoning"              # 时间推理
    KNOWLEDGE_UPDATES = "knowledge_updates"                # 知识更新
    ABSTENTION = "abstention"                              # 识别未知


class EventType(str, Enum):
    """事件类型"""
    CHAT = "chat"
    PERCEPTION = "perception"
    THOUGHT = "thought"
    REFLECTION = "reflection"
    EXPERIENCE = "experience"


class MemoryScope(str, Enum):
    """记忆范围"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    FULL = "full"


# ============== 请求模型 ==============

class AddMemoryRequest(BaseModel):
    """
    添加记忆请求
    
    对应能力: Knowledge Updates (KU)
    """
    content: str = Field(..., description="记忆内容")
    event_type: EventType = Field(..., description="事件类型")
    actor: str = Field(..., description="执行者标识")
    audience: Optional[List[str]] = Field(default=None, description="受众列表")
    impact_score: float = Field(default=0.5, ge=0.0, le=1.0, description="冲击力评分")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")
    timestamp: Optional[datetime] = Field(default=None, description="时间戳（默认当前时间）")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "用户告诉我他喜欢蓝色",
                "event_type": "chat",
                "actor": "user",
                "audience": ["assistant"],
                "impact_score": 0.7,
                "session_id": "session_123"
            }
        }


class RetrieveMemoryRequest(BaseModel):
    """
    检索记忆请求
    
    对应能力: Information Extraction (IE), Multi-Session Reasoning (MR)
    """
    query: str = Field(..., description="检索查询")
    top_k: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="相似度阈值")
    memory_scope: MemoryScope = Field(default=MemoryScope.FULL, description="检索范围")
    audience: Optional[List[str]] = Field(default=None, description="限定受众")
    time_range: Optional[Dict[str, datetime]] = Field(default=None, description="时间范围")
    required_capabilities: Optional[List[MemoryCapability]] = Field(
        default=None,
        description="要求的能力类型"
    )
    enable_temporal_reasoning: bool = Field(default=False, description="启用时间推理")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "用户喜欢什么颜色？",
                "top_k": 5,
                "threshold": 0.3,
                "memory_scope": "full"
            }
        }


class UpdateMemoryRequest(BaseModel):
    """
    更新记忆请求
    
    对应能力: Knowledge Updates (KU)
    """
    event_id: UUID1 = Field(..., description="要更新的记忆 ID")
    new_content: Optional[str] = Field(default=None, description="新内容（可选）")
    new_impact_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="新冲击力")
    conflict_resolution: Literal["keep_new", "keep_old", "merge"] = Field(
        default="merge",
        description="冲突解决策略"
    )
    reason: Optional[str] = Field(default=None, description="更新原因")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "uuid-123",
                "new_content": "用户现在说他喜欢绿色",
                "new_impact_score": 0.8,
                "conflict_resolution": "merge"
            }
        }


class DeleteMemoryRequest(BaseModel):
    """删除记忆请求"""
    event_id: UUID1 = Field(..., description="要删除的记忆 ID")
    hard_delete: bool = Field(default=False, description="硬删除（否则为软删除/遗忘）")


class BatchOperationRequest(BaseModel):
    """批量操作请求"""
    operations: List[Dict[str, Any]] = Field(..., description="操作列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operations": [
                    {"type": "add", "content": "...", "event_type": "chat"},
                    {"type": "update", "event_id": "uuid-123", "new_content": "..."},
                ]
            }
        }


class QueryRequest(BaseModel):
    """
    复杂查询请求
    
    对应能力: Temporal Reasoning (TR)
    """
    question: str = Field(..., description="自然语言问题")
    context_window: Optional[int] = Field(default=None, description="上下文窗口大小")
    require_temporal_order: bool = Field(default=True, description="是否要求时间顺序")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "用户上周三做了什么？",
                "require_temporal_order": True
            }
        }


# ============== 响应模型 ==============

class MemoryMetadata(BaseModel):
    """记忆元数据"""
    event_id: UUID1
    event_type: EventType
    actor: str
    audience: Optional[List[str]]
    impact_score: float
    strength: float
    decay_rate: float
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    session_id: Optional[str] = None


class MemoryItem(BaseModel):
    """记忆项"""
    event_id: UUID1
    content: str
    metadata: MemoryMetadata
    relevance_score: float = Field(..., description="相关性评分")
    matched_capabilities: Optional[List[MemoryCapability]] = Field(
        default=None,
        description="匹配的能力类型"
    )


class RetrieveMemoryResponse(BaseModel):
    """
    检索记忆响应
    
    对应能力: Information Extraction (IE)
    """
    results: List[MemoryItem] = Field(..., description="检索结果")
    total_count: int = Field(..., description="符合条件总数")
    query: str = Field(..., description="原始查询")
    session_id: Optional[str] = Field(default=None)
    capabilities_detected: List[MemoryCapability] = Field(
        default_factory=list,
        description="检测到的能力类型"
    )
    abstention: bool = Field(default=False, description="是否识别为未知")
    abstention_reason: Optional[str] = Field(default=None, description="未知原因")


class AddMemoryResponse(BaseModel):
    """添加记忆响应"""
    event_id: UUID1
    success: bool
    message: str
    decay_rate: float
    estimated_lifespan_days: Optional[float] = Field(
        default=None,
        description="预估寿命（天）"
    )


class UpdateMemoryResponse(BaseModel):
    """更新记忆响应"""
    event_id: UUID1
    success: bool
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    conflict_detected: bool = False
    conflict_resolved_with: Optional[str] = None
    message: str


class QueryResponse(BaseModel):
    """
    查询响应
    
    对应能力: Temporal Reasoning (TR), Multi-Session Reasoning (MR)
    """
    answer: str = Field(..., description="答案")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    supporting_memories: List[MemoryItem] = Field(..., description="支持记忆")
    temporal_ordered: bool = Field(default=False, description="是否按时间排序")
    reasoning_chain: Optional[List[str]] = Field(
        default=None,
        description="推理链"
    )
    abstention: bool = Field(default=False)
    abstention_reason: Optional[str] = None


class BatchOperationResponse(BaseModel):
    """批量操作响应"""
    results: List[Dict[str, Any]]
    success_count: int
    failure_count: int
    errors: List[str] = Field(default_factory=list)


class SystemStatus(BaseModel):
    """系统状态"""
    total_memories: int
    short_term_memories: int
    long_term_memories: int
    avg_strength: float
    avg_decay_rate: float
    elo_statistics: Dict[str, float]
    capability_coverage: Dict[MemoryCapability, float] = Field(
        default_factory=dict,
        description="各能力覆盖度"
    )


# ============== 接口函数定义 ==============

# 以下函数签名供实现参考

def create_memory_api_spec() -> Dict[str, Any]:
    """
    创建 API 规范
    
    返回 OpenAPI 兼容的 schema
    """
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Engram Memory API",
            "description": "LongMemEval 兼容的记忆系统 API",
            "version": "1.0.0"
        },
        "paths": {
            "/memory": {
                "post": {
                    "summary": "添加记忆",
                    "operationId": "add_memory",
                    "requestBody": AddMemoryRequest.model_json_schema(),
                    "responses": {
                        "200": AddMemoryResponse.model_json_schema()
                    }
                }
            },
            "/memory/retrieve": {
                "post": {
                    "summary": "检索记忆",
                    "operationId": "retrieve_memory",
                    "requestBody": RetrieveMemoryRequest.model_json_schema(),
                    "responses": {
                        "200": RetrieveMemoryResponse.model_json_schema()
                    }
                }
            },
            "/memory/{event_id}": {
                "put": {
                    "summary": "更新记忆",
                    "operationId": "update_memory",
                    "requestBody": UpdateMemoryRequest.model_json_schema()
                },
                "delete": {
                    "summary": "删除记忆",
                    "operationId": "delete_memory"
                }
            },
            "/memory/query": {
                "post": {
                    "summary": "复杂查询（支持时间推理）",
                    "operationId": "query_memory",
                    "requestBody": QueryRequest.model_json_schema(),
                    "responses": {
                        "200": QueryResponse.model_json_schema()
                    }
                }
            },
            "/memory/batch": {
                "post": {
                    "summary": "批量操作",
                    "operationId": "batch_operations",
                    "requestBody": BatchOperationRequest.model_json_schema(),
                    "responses": {
                        "200": BatchOperationResponse.model_json_schema()
                    }
                }
            },
            "/status": {
                "get": {
                    "summary": "获取系统状态",
                    "operationId": "get_status",
                    "responses": {
                        "200": SystemStatus.model_json_schema()
                    }
                }
            }
        }
    }


# ============== 能力映射 ==============

CAPABILITY_EVENT_TYPE_MAP: Dict[MemoryCapability, List[EventType]] = {
    MemoryCapability.INFORMATION_EXTRACTION: [
        EventType.CHAT,
        EventType.PERCEPTION,
    ],
    MemoryCapability.MULTI_SESSION_REASONING: [
        EventType.THOUGHT,
        EventType.REFLECTION,
    ],
    MemoryCapability.TEMPORAL_REASONING: [
        EventType.EXPERIENCE,
        EventType.REFLECTION,
    ],
    MemoryCapability.KNOWLEDGE_UPDATES: [
        EventType.CHAT,
        EventType.THOUGHT,
    ],
    MemoryCapability.ABSTENTION: [],  # 特殊标记
}


def infer_capabilities(event_types: List[EventType]) -> List[MemoryCapability]:
    """根据事件类型推断可支持的能力"""
    capabilities = set()
    for et in event_types:
        for cap, types in CAPABILITY_EVENT_TYPE_MAP.items():
            if et in types:
                capabilities.add(cap)
    return list(capabilities)


# ============== 测试用例 ==============

def generate_test_cases() -> Dict[str, Any]:
    """生成测试用例"""
    return {
        "information_extraction": {
            "query": "用户叫什么名字？",
            "expected_capabilities": [MemoryCapability.INFORMATION_EXTRACTION],
        },
        "multi_session": {
            "query": "用户这周和上周的工作内容有什么变化？",
            "expected_capabilities": [MemoryCapability.MULTI_SESSION_REASONING],
        },
        "temporal": {
            "query": "用户是什么时候开始喜欢蓝色的？",
            "expected_capabilities": [MemoryCapability.TEMPORAL_REASONING],
        },
        "knowledge_update": {
            "add_request": {
                "content": "用户更正说其实喜欢绿色",
                "event_type": "chat",
                "actor": "user",
                "impact_score": 0.8,
            },
            "expected_capabilities": [MemoryCapability.KNOWLEDGE_UPDATES],
        },
        "abstention": {
            "query": "用户的身份证号码是多少？",
            "expected_abstention": True,
        },
    }
