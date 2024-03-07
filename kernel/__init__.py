"""Not using Semantic Kernel for maximum flexibility. Semantic Kernel does not support system message currently."""

from pydantic import BaseModel, Field
from typing import Union, List, Literal, Dict, Any
from text2vec import SentenceModel
from abc import ABC, abstractmethod
from settings import get_settings
from utils.template import DEFAULT_PROMPT
import openai

setting = get_settings()


class ChatService(BaseModel, ABC):
    service_type: Literal["Base"] = "Base"

    @abstractmethod
    def get_response(self, query: str, **kwargs) -> str:
        pass


class ChatGPTChatService(ChatService):
    service_type: Literal["ChatGPT"] = "ChatGPT"
    api_key: str = Field(setting.OPENAI_KEY, exclude=True)
    deployment: str = setting.AZURE_OPENAI_DEPLOYMENT
    endpoint: str = Field(setting.AZURE_ENDPOINT, exclude=True)
    api_version: str = "2023-03-15-preview"

    def set_openai(self):
        openai.api_type = "azure"
        openai.api_key = self.api_key
        openai.api_base = self.endpoint
        openai.api_version = self.api_version

    async def get_response(
        self,
        query: str,
        instruction: str = DEFAULT_PROMPT,
        history: List[Dict[str, str]] = [],
        **kwargs
    ) -> str:
        self.set_openai()
        messages = [{"role": "system", "content": instruction}] if instruction else []
        if query is not None:
            messages += history + [{"role": "user", "content": query}]
        response = await openai.ChatCompletion.acreate(
            messages=messages, engine=self.deployment, **kwargs
        )

        return response["choices"][0]["message"]["content"]


class EmbeddingService(BaseModel, ABC):
    service_type: Literal["Base"] = "Base"
    embedding_model: Any = Field(exclude=True)

    @abstractmethod
    def get_sentence_embeddings(self, sentences: Union[str, List[str]]):
        pass

    class Config:
        arbitrary_types_allowed = True


class Text2VecEmbeddingService(EmbeddingService):
    service_type: Literal["Text2Vec"] = "Text2Vec"
    embedding_model: SentenceModel = Field(
        SentenceModel(setting.TEXT2VEC_PATH), exclude=True
    )

    def get_sentence_embeddings(self, sentences: str | List[str]):
        return self.embedding_model.encode(sentences)


class Kernel(BaseModel):
    chat_service: ChatService = Field(ChatGPTChatService(), exclude=True)
    embedding_service: EmbeddingService = Field(
        Text2VecEmbeddingService(), exclude=True
    )

    async def get_response(
        self,
        query: str,
        instruction: str = DEFAULT_PROMPT,
        history: List[Dict[str, str]] = [],
        **kwargs
    ) -> str:
        return await self.chat_service.get_response(
            query, instruction, history, **kwargs
        )

    def get_sentence_embeddings(self, sentences: str | List[str]):
        return self.embedding_service.get_sentence_embeddings(sentences=sentences)

    class Config:
        arbitrary_types_allowed = True
