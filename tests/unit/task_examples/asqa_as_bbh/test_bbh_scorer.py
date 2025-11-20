import pandas as pd
import pytest

from prompt_optimizer.chat_model import StubChatModel
from prompt_optimizer.chat_prompt import FStringChatTemplate
from prompt_optimizer.instruction import Instruction
from task_examples.asqa_as_bbh.bbh_task import BBHTask


def test__bbh_task__raises__when_placeholders_missing():
    with pytest.raises(ValueError):
        _ = BBHTask(
            task_solver_llm=StubChatModel(["stub response"]),
            prompt_template=FStringChatTemplate(template_messages=[("system", "System prompt")]),
            data_df=pd.DataFrame({"input": ["Sample input"]}),
            num_samples_to_score=10,
        )


@pytest.mark.parametrize(
    "stub_response, expected_score", [(["Yes", "No"], 1.0), (["No", "No"], 0.5), (["No", "Yes"], 0.0)]
)
def test__bbh_task__score(stub_response: list[str], expected_score: float) -> None:
    data_df = pd.DataFrame(
        {
            "input": [
                "Is the sky blue?",
                "Is fire cold?",
            ],
            "target": [
                "Yes",
                "No",
            ],
        }
    )

    task = BBHTask(
        task_solver_llm=StubChatModel(stub_response),
        prompt_template=FStringChatTemplate(
            template_messages=[
                ("system", "{INSTRUCTION}"),
                ("user", "{TASK_SAMPLE}"),
            ]
        ),
        data_df=data_df,
    )

    score = task.score(Instruction("Answer with 'Yes' or 'No'."))
    assert score.score == expected_score
