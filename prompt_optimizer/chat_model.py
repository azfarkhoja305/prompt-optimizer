from abc import ABC, abstractmethod
from typing import Sequence

from langchain_core.language_models import BaseChatModel as LangchainChatClient
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage as LangchainHumanMessage
from langchain_core.messages import SystemMessage as LangchainSystemMessage
from langchain_core.messages import AIMessage as LangchainAIMessage

from prompt_optimizer.chat_prompt import ChatMessage, AIMessage, SystemMessage, UserMessage


def chat_messages_to_lc_message(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, UserMessage):
            lc_messages.append(LangchainHumanMessage(content=message.content))
        elif isinstance(message, SystemMessage):
            lc_messages.append(LangchainSystemMessage(content=message.content))
        elif isinstance(message, AIMessage):
            lc_messages.append(LangchainAIMessage(content=message.content))
        else:
            raise ValueError(f"Unknown chat role: {message.role}")
    return lc_messages


class ChatModel(ABC):
    @abstractmethod
    def response(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        pass

    @abstractmethod
    def batch_response(self, messages: Sequence[Sequence[ChatMessage]], max_tokens: int) -> list[str]:
        pass


class LangchainChatModel(ChatModel):
    def __init__(self, langchain_client: LangchainChatClient, temperature: float = 0) -> None:
        self._chat_client = langchain_client
        self._temperature = temperature

    def response(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        lc_messages = chat_messages_to_lc_message(messages)
        response = self._chat_client.invoke(lc_messages, max_tokens=max_tokens, temperature=self._temperature).text()
        return response

    def batch_response(self, messages: Sequence[Sequence[ChatMessage]], max_tokens: int) -> list[str]:
        batch_lc_messages = [chat_messages_to_lc_message(msg) for msg in messages]
        responses = self._chat_client.batch(batch_lc_messages, max_tokens=max_tokens, temperature=self._temperature)  # type: ignore[arg-type]
        return [response.text() for response in responses]


class StubChatModel(ChatModel):
    def __init__(self, stub_response: list[str]) -> None:
        self._stub_response = stub_response

    def response(self, messages: Sequence[ChatMessage], max_tokens: int) -> str:
        return self._stub_response[0]

    def batch_response(self, messages: Sequence[Sequence[ChatMessage]], max_tokens: int) -> list[str]:
        return self._stub_response
