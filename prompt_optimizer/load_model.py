import os
from dataclasses import dataclass
from enum import Enum, auto

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_openai import AzureChatOpenAI

from prompt_optimizer.chat_model import ChatModel, LangchainChatModel, StubChatModel


class ModelEnum(Enum):
    pass


class OpenAIModel(ModelEnum):
    GPT_4O = auto()
    GPT_4O_MINI = auto()
    GPT_41 = auto()
    GPT_41_MINI = auto()


class BedrockModel(ModelEnum):
    NOVA_LITE = auto()
    NOVA_PRO = auto()


class StubModel(ModelEnum):
    COUNT_STUB = auto()


@dataclass
class ModelProperty:
    model_name: str
    model_version: str
    model_provider: str

    @property
    def model_id(self):
        return self.model_name + "-" + self.model_version


OPENAI_MODELS_PROPERTY_MAPPINGS = {
    OpenAIModel.GPT_4O: ModelProperty(model_name="gpt-4o", model_version="2024-08-06", model_provider="openai"),
    OpenAIModel.GPT_4O_MINI: ModelProperty(
        model_name="gpt-4o-mini", model_version="2024-07-18", model_provider="openai"
    ),
    OpenAIModel.GPT_41: ModelProperty(model_name="gpt-4.1", model_version="2025-04-14", model_provider="openai"),
    OpenAIModel.GPT_41_MINI: ModelProperty(
        model_name="gpt-4.1-mini", model_version="2025-04-14", model_provider="openai"
    ),
}

BEDROCK_MODELS_PROPERTY_MAPPINGS = {
    BedrockModel.NOVA_LITE: ModelProperty(model_name="amazon.nova-lite", model_version="v1_0", model_provider="amazon"),
    BedrockModel.NOVA_PRO: ModelProperty(model_name="amazon.nova-pro", model_version="v1_0", model_provider="amazon"),
}


def load_openai_model(model: OpenAIModel) -> LangchainChatModel:
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    model_property = OPENAI_MODELS_PROPERTY_MAPPINGS[model]

    azure_openai_client = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,  # type: ignore
        model=model_property.model_name,
        model_version=model_property.model_version,
        default_headers={"model-version": model_property.model_version},
    )
    return LangchainChatModel(azure_openai_client)


def load_bedrock_model(model: BedrockModel) -> LangchainChatModel:
    model_property = BEDROCK_MODELS_PROPERTY_MAPPINGS[model]
    inference_profile = os.environ[str.upper(model_property.model_id)]
    bedrock_client = ChatBedrockConverse(
        client=boto3.client("bedrock-runtime"),
        model=inference_profile,
        base_model=model_property.model_name,
        provider=model_property.model_provider,
    )
    return LangchainChatModel(bedrock_client)


def load_stub_model(model: StubModel) -> ChatModel:
    if model == StubModel.COUNT_STUB:
        return StubChatModel(stub_response=["1,2,3,4,5", "6,7,8,9"])


def load_model(model: ModelEnum) -> ChatModel:
    if isinstance(model, OpenAIModel):
        return load_openai_model(model)
    elif isinstance(model, BedrockModel):
        return load_bedrock_model(model)
    elif isinstance(model, StubModel):
        return load_stub_model(model)
    else:
        raise ValueError(f"Unknown model type: {model}")
