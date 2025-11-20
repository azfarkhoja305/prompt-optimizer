import pandas as pd

from prompt_optimizer.chat_model import ChatModel
from prompt_optimizer.chat_prompt import ChatTemplate
from prompt_optimizer.instruction import Instruction, InstructionScore, Task


class BBHTask(Task):
    INSTRUCTION = "INSTRUCTION"
    TASK_SAMPLE = "TASK_SAMPLE"
    REQUIRED_PLACEHOLDERS = {INSTRUCTION, TASK_SAMPLE}

    def __init__(
        self,
        task_solver_llm: ChatModel,
        prompt_template: ChatTemplate,
        data_df: pd.DataFrame,
        num_samples_to_score: int | None = None,
    ) -> None:
        self._task_solver_llm = task_solver_llm
        self._prompt_template = prompt_template
        self._data_df = data_df if num_samples_to_score is None else data_df.iloc[:num_samples_to_score]

        if missing := self.REQUIRED_PLACEHOLDERS - self._prompt_template.placeholders:
            raise ValueError(f"Prompt template is missing required placeholders: {missing}")

    def score(self, instruction: Instruction) -> InstructionScore:
        task_samples, targets = self._data_df["input"].tolist(), self._data_df["target"].tolist()

        batched_messages = [
            self._prompt_template.fill_template(
                {
                    self.INSTRUCTION: instruction.instruction,
                    self.TASK_SAMPLE: sample,
                }
            )
            for sample in task_samples
        ]

        responses = self._task_solver_llm.batch_response(batched_messages, max_tokens=5)

        score_sum = sum([label.lower() in pred.lower() for pred, label in zip(responses, targets, strict=True)])
        score = score_sum / len(targets)
        return InstructionScore(score=score)
