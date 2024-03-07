from pydantic import BaseModel
from typing import List
from enum import Enum
from utils.message import SimpleChatSession, ChatMessage


class ServiceStatueCode(Enum):
    success = 1
    failed_need_retry = 2
    failed_no_need_retry = 3


class ServiceBaseResponse(BaseModel):
    code: ServiceStatueCode = ServiceStatueCode.success
    msg: str = ""


class GetChatListRequest(BaseModel):
    include: List[str]


class GetChatListResponse(ServiceBaseResponse):
    data: List[SimpleChatSession]


class GetChatRequest(BaseModel):
    chat_id: str


class GetChatResponse(ServiceBaseResponse):
    data: SimpleChatSession


class CreateChatRequest(BaseModel):
    member: List[str]


class CreateChatResponse(ServiceBaseResponse):
    data: SimpleChatSession


class UpdateChatRequest(BaseModel):
    chat: SimpleChatSession


class UpdateChatResponse(BaseModel):
    data: SimpleChatSession


class ChatEventRequest(BaseModel):
    message: ChatMessage


class ChatEventResponse(ServiceBaseResponse):
    data: ChatMessage
