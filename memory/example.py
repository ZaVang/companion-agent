"""一个Basic Memory Example，用来演示向下（对EventStream）和向上（对Retrieval）的API形式"""
from pydantic import BaseModel, Field
from .event import EventStream
from abc import ABC, abstractmethod


class BaseMemory(BaseModel, ABC):
    agent_id: str
    world_stream: EventStream = Field(
        EventStream(), exclude=True, description="世界完整事件线，序列化不会带"
    )
    personal_stream: EventStream = Field(
        EventStream(),
        exclude=False,
        description="个人事件线，可以用可见过滤从world_stream中validate出来。序列化会带",
    )

    @abstractmethod
    def retrieval(self, query: str, filter, **kwargs):
        pass


class DemoMemory(BaseMemory):
    def retrieval(self, query: str, filter, top_k=5, **kwargs):
        pass

    def chat_retrieval(self, query: str, sender: str, top_k=5, **kwargs):
        """针对不同事件类型触发的retrieval，可以分别写retrieval逻辑。"""
        pass
