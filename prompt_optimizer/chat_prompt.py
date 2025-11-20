import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence, Literal

RoleStr = Literal["system", "user", "ai"]


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    AI = "ai"


@dataclass(frozen=True)
class ChatMessage(ABC):
    role: ClassVar[ChatRole]
    content: str

    @classmethod
    def from_role_and_content_strings(cls, role: str, content: str) -> "ChatMessage":
        chat_role = ChatRole(role)
        if chat_role == ChatRole.SYSTEM:
            return SystemMessage(content=content)
        elif chat_role == ChatRole.USER:
            return UserMessage(content=content)
        elif chat_role == ChatRole.AI:
            return AIMessage(content=content)
        else:
            raise ValueError(f"Unknown chat role: {chat_role}")


@dataclass(frozen=True)
class SystemMessage(ChatMessage):
    role: ClassVar[ChatRole] = ChatRole.SYSTEM


@dataclass(frozen=True)
class UserMessage(ChatMessage):
    role: ClassVar[ChatRole] = ChatRole.USER


@dataclass(frozen=True)
class AIMessage(ChatMessage):
    role: ClassVar[ChatRole] = ChatRole.AI


class ChatTemplate(ABC):
    @property
    @abstractmethod
    def placeholders(self) -> set[str]:
        pass

    @abstractmethod
    def fill_template(self, values: dict[str, str]) -> list[ChatMessage]:
        pass


class FStringChatTemplate(ChatTemplate):
    def __init__(self, template_messages: Sequence[tuple[RoleStr, str]]) -> None:
        self._template_messages = template_messages
        self._placeholders: set | None = None

    @property
    def placeholders(self) -> set[str]:
        if self._placeholders is None:
            self._placeholders = self._extract_placeholders([message for _, message in self._template_messages])
        return self._placeholders

    def fill_template(self, values: dict[str, str]) -> list[ChatMessage]:
        if missing := self.placeholders - values.keys():
            raise ValueError(f"Missing values for placeholders: {missing}")

        filled_messages: list[ChatMessage] = []
        for role_str, template in self._template_messages:
            filled_content = template.format(**values)
            filled_messages.append(ChatMessage.from_role_and_content_strings(role_str, filled_content))
        return filled_messages

    def _extract_placeholders(self, messages: list[str]) -> set[str]:
        placeholders = set()
        for message in messages:
            fields = [field for _, field, _, _ in string.Formatter().parse(message) if field]
            placeholders.update(fields)
        return placeholders
