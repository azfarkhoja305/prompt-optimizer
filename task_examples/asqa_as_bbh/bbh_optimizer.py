from datetime import datetime

import os

import pandas as pd

from prompt_optimizer.chat_model import ChatModel
from prompt_optimizer.chat_prompt import FStringChatTemplate, ChatTemplate
from prompt_optimizer.load_model import load_model, OpenAIModel, BedrockModel
from prompt_optimizer.opro import LLMInstructionGenerator, OproOptimizer
from prompt_optimizer.instruction import Instruction
from bbh_task import BBHTask
from bbh_prompt import META_SYSTEM_PROMPT, TASK_SOLVER_SYSTEM_PROMPT, TASK_SOLVER_USER_PROMPT
from bbh_exemplar import BBHExemplar
from dotenv import load_dotenv

from task_examples.setup_logging import setup_logging

FILE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(FILE_PATH, "dataset")

TRAIN_FILE = os.path.join(DATASET_PATH, "train.csv")
VALID_FILE = os.path.join(DATASET_PATH, "valid.csv")

ts = datetime.now().strftime("%Y%m%d%H%M%S")
SAVE_FILE = os.path.join(DATASET_PATH, f"bbh_optimized_prompts_{ts}.pkl")


def bootstrap() -> tuple[LLMInstructionGenerator, BBHTask, BBHTask]:
    optimizer_llm = load_model(OpenAIModel.GPT_41)
    task_solver_llm = load_model(BedrockModel.NOVA_LITE)
    optimizer_prompt_template = FStringChatTemplate(
        template_messages=[("system", META_SYSTEM_PROMPT)],
    )
    solver_prompt_template = FStringChatTemplate(
        template_messages=[("system", TASK_SOLVER_SYSTEM_PROMPT), ("user", TASK_SOLVER_USER_PROMPT)],
    )
    exemplar_formatter = BBHExemplar(data_df=pd.read_csv(TRAIN_FILE), num_samples=5)

    llm_instruction_generator = create_llm_instruction_generator(
        optimizer_llm=optimizer_llm,
        prompt_template=optimizer_prompt_template,
        exemplar_formatter=exemplar_formatter,
    )

    train_task = create_task(
        task_llm=task_solver_llm,
        prompt_template=solver_prompt_template,
        data_file=TRAIN_FILE,
        num_samples_to_score=10
    )

    valid_task = create_task(
        task_llm=task_solver_llm,
        prompt_template=solver_prompt_template,
        data_file=VALID_FILE,
        num_samples_to_score=10
    )

    return llm_instruction_generator, train_task, valid_task


def create_llm_instruction_generator(
    optimizer_llm: ChatModel, prompt_template: ChatTemplate, exemplar_formatter: BBHExemplar
) -> LLMInstructionGenerator:
    return LLMInstructionGenerator(
        prompt_optimizer_llm=optimizer_llm,
        optimizer_prompt_template=prompt_template,
        task_exemplars_fn=exemplar_formatter,
    )


def create_task(
    task_llm: ChatModel, prompt_template: ChatTemplate, data_file: str, num_samples_to_score: int | None = None
) -> BBHTask:
    return BBHTask(
        task_solver_llm=task_llm,
        prompt_template=prompt_template,
        data_df=pd.read_csv(data_file),
        num_samples_to_score=num_samples_to_score,
    )


def main() -> None:
    instruction_generator, train_task, valid_task = bootstrap()
    optimizer = OproOptimizer(
        instruction_generator=instruction_generator,
        train_task=train_task,
        validation_task=valid_task,
        num_train_iterations=3,
    )
    optimized_prompt_instructions = optimizer.run(seed_instructions=[Instruction("")])
    optimized_prompt_instructions.save(SAVE_FILE)


if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    main()
