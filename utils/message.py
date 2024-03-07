from pydantic import BaseModel, Field, UUID1
from utils.schema import DateTime
from typing import Any, List, Optional
import uuid
from enum import Enum


class FeishuIdentity(BaseModel):
    open_id: str
    chat_id: str
    app_id: str = Field(None, description="Agent一般通过app形式在飞书体现")


class ChannelIdentity(BaseModel):
    """系统支持的所有Channel"""

    feishu_identity: Optional[FeishuIdentity] = Field(None)


class Identity(BaseModel):
    id: UUID1 = Field(
        default_factory=uuid.uuid1, description="唯一ID，与Agent Persona的ID一致"
    )


class DetailItentity(Identity):
    name: str = Field(None, description="名字")
    is_agent: bool = Field(None, description="是否为AI Agent，否则为真人")
    channel_identity: ChannelIdentity = Field(
        ChannelIdentity(),
        description="此身份在不同通信系统中的对应身份，例如飞书、微信等。这个应该是个计算字段，从数据库取回",
    )


class MessageChannel(str, Enum):
    feishu = "feishu"
    aic_web = "aic_web"


class MessageContextMetadata(BaseModel):
    """消息的Context（包括用的对话系统、私聊或哪个群聊）信息，不包括具体的对话上下文"""

    channel: MessageChannel
    chat_id: str = Field(
        ..., description="飞书可以用ChatID定位。如果有的系统对单聊群聊使用不同ID，可以通过前缀或者补充字段区分"
    )


class BaseMessage(BaseModel):
    sender: Identity
    receiver: Identity | List[Identity]
    content: Any
    event_id: UUID1 = Field(default_factory=uuid.uuid1)
    create_time: DateTime
    context: MessageContextMetadata
    metadata: dict = Field({}, description="一些额外信息")


from utils.schema import MessageContent


class ChatMessage(BaseMessage):
    content: MessageContent


class BaseChatSession(BaseModel):
    chat_id: UUID1 = Field(default_factory=uuid.uuid1)
    member: List[Identity]
    event_ids: List[UUID1] = []
    events: List[Any] = Field([], exclude=True)


class SimpleChatSession(BaseChatSession):
    events: List[ChatMessage] = Field(
        [], exclude=False, description="目前是ChatSession。信息更多的应该是EventStream里的event类型"
    )
