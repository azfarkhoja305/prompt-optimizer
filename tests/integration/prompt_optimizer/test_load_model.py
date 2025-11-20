import pytest

from prompt_optimizer.chat_model import UserMessage
from prompt_optimizer.load_model import BedrockModel, OpenAIModel, StubModel, load_model


@pytest.mark.parametrize("model", [OpenAIModel.GPT_41_MINI, BedrockModel.NOVA_LITE, StubModel.COUNT_STUB])
def test_load_openai_model(model) -> None:
    chat_model = load_model(model)
    response = chat_model.response([UserMessage("Count from 1 to 5")], max_tokens=100)
    assert all(i in response for i in ["1", "2", "3", "4", "5"])

    responses = chat_model.batch_response(
        [
            [UserMessage("Count from 1 to 5")],
            [UserMessage("Count from 6 to 9")],
        ],
        max_tokens=100,
    )
    assert all(i in responses[0] for i in ["1", "2", "3", "4", "5"])
    assert all(i in responses[1] for i in ["6", "7", "8", "9"])
