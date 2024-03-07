from pydantic import BaseModel, Field, model_validator, UUID1
from abc import ABC, abstractmethod
from utils.message import SimpleChatSession, ChatMessage, Identity, DetailItentity
from typing import List
import uuid


class AICChatStoreBase(BaseModel, ABC):
    @abstractmethod
    def get_chat_list(self):
        pass

    @abstractmethod
    def get_chat(self):
        pass

    @abstractmethod
    def create_chat(self):
        pass

    @abstractmethod
    def update_chat(self):
        pass


class RecordNotFoundError(Exception):
    """数据库没找到记录"""


class RecordAlreadyExistError(Exception):
    """数据库已有记录"""


import json

from pydantic import RootModel


class SessionList(RootModel):
    root: List[SimpleChatSession]


class EventList(RootModel):
    root: List[ChatMessage]


class IdentityList(RootModel):
    root: List[DetailItentity]


class AICChatStoreLocal(AICChatStoreBase):
    """临时用一下，得移到MongoDB"""

    db_directory: str = "db"
    chat_path: str = "chat"
    event_collection: str = "chat_event"
    session_collection: str = "chat_session"
    identity_collection: str = "chat_identity"
    sessions: SessionList = Field(SessionList([]), exclude=True)
    events: EventList = Field(EventList([]), exclude=True)
    identities: IdentityList = Field(IdentityList([]), exclude=True)

    @model_validator(mode="after")
    def load_data(self) -> "AICChatStoreLocal":
        with open(f"{self.db_path}/{self.session_collection}.json", "r") as f:
            sessions = json.load(f)
        if sessions:
            self.sessions = SessionList(sessions)

        with open(f"{self.db_path}/{self.event_collection}.json", "r") as f:
            events = json.load(f)
        if events:
            self.events = EventList(events)

        with open(f"{self.db_path}/{self.identity_collection}.json", "r") as f:
            identities = json.load(f)
        if identities:
            self.identities = IdentityList(identities)
        return self

    def dump_data(
        self, sessions: bool = False, events: bool = False, identities: bool = False
    ) -> None:
        if sessions:
            with open(f"{self.db_path}/{self.session_collection}.json", "w") as f:
                f.write(self.sessions.model_dump_json())
        if events:
            with open(f"{self.db_path}/{self.event_collection}.json", "w") as f:
                f.write(self.events.model_dump_json())
        if identities:
            with open(f"{self.db_path}/{self.identity_collection}.json", "w") as f:
                f.write(self.identities.model_dump_json())

    @property
    def db_path(self) -> str:
        return f"{self.db_directory}/{self.chat_path}"

    def get_event(self, event_id: UUID1) -> ChatMessage:
        try:
            event = [e for e in self.events.root if e.event_id == event_id][0]
            return event
        except:
            raise RecordNotFoundError(f"找不到此event_id:{event_id}对应的消息记录。")

    def get_chat_list(self, include: List[str]) -> List[SimpleChatSession]:
        include = [uuid.UUID(i) for i in include]
        return [
            s
            for s in self.sessions.root
            if set(include).issubset(set([i.id for i in s.member]))
        ]

    def get_chat(self, chat_id: str) -> "SimpleChatSession":
        chat_id = uuid.UUID(chat_id)
        try:
            chat = [s for s in self.sessions.root if s.chat_id == chat_id][0]
            chat.events = [self.get_event(event_id=eid) for eid in chat.event_ids]
        except:
            raise RecordNotFoundError(f"找不到此chat_id:{chat_id}对应的session。")
        return chat

    def create_chat(self, member: List[str]) -> SimpleChatSession:
        chat = self.get_chat_list(include=member)
        if chat:
            raise RecordAlreadyExistError(
                f"已存在此成员的chat:{member}，chat_id:{chat[0].chat_id}"
            )
        else:
            chat = SimpleChatSession(member=[{"id": _id} for _id in member])
            self.sessions.root.append(chat)
            self.dump_data(sessions=True)
            return chat

    def update_chat(self, chat: SimpleChatSession) -> SimpleChatSession:
        for index, _chat in enumerate(self.sessions.root):
            if _chat.chat_id == chat.chat_id:
                self.sessions.root[index] = chat
                self.dump_data(sessions=True)
                return chat
        raise RecordNotFoundError(f"未找到此chat_id的记录{chat.chat_id}，无法更新")

    def add_event(self, event: ChatMessage) -> ChatMessage:
        self.events.root.append(event)
        self.dump_data(events=True)
        return event

    def add_identity(self, identity: DetailItentity) -> DetailItentity:
        self.identities.root.append(identity)
        self.dump_data(identities=True)
        return identity

    def get_identity(self, id: UUID1) -> DetailItentity:
        try:
            identity = [i for i in self.identities.root if i.id == id][0]
            return identity
        except:
            raise RecordNotFoundError(f"找不到此id:{id.hex}对应的身份信息。")

    def update_identity(self, identity: DetailItentity) -> DetailItentity:
        for index, _identity in enumerate(self.identities.root):
            if _identity.id == identity.id:
                self.identities.root[index] = identity
                self.dump_data(identities=True)
                return identity
        raise RecordNotFoundError(f"未找到此identity的记录{identity.id}，无法更新")
