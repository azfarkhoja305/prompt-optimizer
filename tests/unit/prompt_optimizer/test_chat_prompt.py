from prompt_optimizer.chat_prompt import (
    ChatMessage,
    ChatRole,
    SystemMessage,
    UserMessage,
    AIMessage,
    FStringChatTemplate,
)
import pytest


@pytest.mark.parametrize("role", ["system", "user", "ai"])
def test__from_role_and_content_strings(role: str) -> None:
    chat_message = ChatMessage.from_role_and_content_strings(role, "Hello, world!")
    if chat_message.role == ChatRole.SYSTEM:
        assert isinstance(chat_message, SystemMessage)
    elif chat_message.role == ChatRole.USER:
        assert isinstance(chat_message, UserMessage)
    elif chat_message.role == ChatRole.AI:
        assert isinstance(chat_message, AIMessage)

    assert chat_message.content == "Hello, world!"


def test__fstring_chat_template__placeholders() -> None:
    chat_template = FStringChatTemplate(
        template_messages=[
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of {country}?"),
            ("ai", "The capital of {country} is {capital}."),
        ]
    )

    assert chat_template.placeholders == {"country", "capital"}


def test__fstring_chat_template__fill_template() -> None:
    chat_template = FStringChatTemplate(
        template_messages=[
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of {country}?"),
            ("ai", "The capital of {country} is {capital}."),
        ]
    )

    filled_messages = chat_template.fill_template({"country": "France", "capital": "Paris"})

    assert filled_messages == [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]


def test__fstring_chat_template__fill_template_raises_when_missing_placeholders() -> None:
    chat_template = FStringChatTemplate(
        template_messages=[
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of {country}?"),
            ("ai", "The capital of {country} is {capital}."),
        ]
    )

    with pytest.raises(ValueError):
        _ = chat_template.fill_template({"country": "France"})
