import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable

from prompt_optimizer.chat_model import ChatModel
from prompt_optimizer.chat_prompt import ChatTemplate
from tqdm import trange, tqdm

from prompt_optimizer.instruction import (
    Instruction,
    InstructionScore,
    OptimizedInstructions,
    Task,
    EarlyStoppingStrategy,
    InstructionGenerator,
)

logger = getLogger(__name__)

ScoredInstruction = tuple[Instruction, InstructionScore]


def extract_instructions_between_tags(response: str) -> list[str]:
    items = [s.strip() for s in re.findall(r"<instruction>(.*?)</instruction>", response, flags=re.DOTALL)]
    return items


def format_scored_instructions(scored_instructions: list[tuple[str, float]]) -> str:
    formatted_instructions = ""
    for idx, (instruction_str, score) in enumerate(scored_instructions):
        formatted_instructions += f"\n{idx + 1}.\nInstruction: {instruction_str}\nScore: {score:.2f}\n"
    return formatted_instructions


class RedundantInstructionStopper:
    def __init__(self, min_each_iteration: int) -> None:
        self._min_each_iteration = min_each_iteration

    def should_stop(self, new_instruction_with_scores: list[ScoredInstruction]) -> bool:
        if len(new_instruction_with_scores) < self._min_each_iteration:
            return True
        return False


@dataclass
class _OproState:
    instructions_at_iteration_n: dict[int, list[Instruction]] = field(default_factory=dict)
    scored_instructions: dict[Instruction, InstructionScore] = field(default_factory=dict)
    validation_scored_instructions: dict[Instruction, InstructionScore] = field(default_factory=dict)

    @property
    def seen_instructions(self) -> set[Instruction]:
        return set(self.scored_instructions.keys())

    def remove_seen_instructions(self, instructions: list[Instruction]) -> list[Instruction]:
        deduped_instructions = set(instructions)
        return [instr for instr in deduped_instructions if instr not in self.seen_instructions]

    def add_new_instructions(self, new_scored_instructions: list[ScoredInstruction]) -> None:
        new_scored_instructions_dict = dict(new_scored_instructions)
        if len(new_scored_instructions_dict) != len(new_scored_instructions):
            logger.warning("Some duplicate instructions were found in the new scored instructions.")
        self.instructions_at_iteration_n[len(self.instructions_at_iteration_n)] = list(
            new_scored_instructions_dict.keys()
        )
        self.scored_instructions.update(new_scored_instructions_dict)

    def top_scored_instructions_with_scores(self, k: int | None = None) -> list[ScoredInstruction]:
        sorted_instructions = sorted(self.scored_instructions.items(), key=lambda item: (item[1].score,), reverse=True)
        if k is None:
            return sorted_instructions
        return sorted_instructions[:k]

    def top_scored_instructions(self, k: int | None = None) -> list[Instruction]:
        top_instructions, _ = zip(*self.top_scored_instructions_with_scores(k))
        return list(top_instructions)

    def add_validation_scored_instructions(self, new_scored_instructions: list[ScoredInstruction]) -> None:
        new_scored_instructions_dict = dict(new_scored_instructions)
        if new_scored_instructions_dict.keys() - self.seen_instructions:
            logger.warning("Validation received some previous unseen instructions. This is not expected.")
        self.validation_scored_instructions.update(new_scored_instructions_dict)

    def create_optimized_instructions(self) -> OptimizedInstructions:
        return OptimizedInstructions(
            instructions_at_iteration_n=self.instructions_at_iteration_n,
            scored_instructions=self.scored_instructions,
            validation_scored_instructions=self.validation_scored_instructions,
        )


def validate_missing_placeholders(required_placeholders: set[str], found_placeholders: set[str]) -> None:
    if missing_placeholders := required_placeholders - found_placeholders:
        raise ValueError(f"Missing placeholders in prompt template: {missing_placeholders}")


class LLMInstructionGenerator(InstructionGenerator):
    EXEMPLARS = "EXEMPLARS"
    SCORED_INSTRUCTIONS = "SCORED_INSTRUCTIONS"
    NUM_NEW_INSTRUCTIONS = "NUM_NEW_INSTRUCTIONS"

    REQUIRED_PLACEHOLDERS = {EXEMPLARS, SCORED_INSTRUCTIONS, NUM_NEW_INSTRUCTIONS}

    def __init__(
        self,
        prompt_optimizer_llm: ChatModel,
        optimizer_prompt_template: ChatTemplate,
        task_exemplars_fn: Callable[[], str],
        instruction_score_formatter_fn: Callable[[list[tuple[str, float]]], str] | None = None,
        instruction_extraction_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._prompt_optimizer_llm = prompt_optimizer_llm
        self._optimizer_prompt_template = optimizer_prompt_template
        self._task_exemplars_fn = task_exemplars_fn
        self._instruction_score_formatter_fn = instruction_score_formatter_fn or format_scored_instructions
        self._instruction_extraction_fn = instruction_extraction_fn or extract_instructions_between_tags

        validate_missing_placeholders(self.REQUIRED_PLACEHOLDERS, optimizer_prompt_template.placeholders)

    def generate_instructions(
        self, prev_top_instructions: list[ScoredInstruction], num_instructions_to_generate: int
    ) -> list[Instruction]:
        formatted_instructions = self._instruction_score_formatter_fn(
            [(instr.instruction, score.score) for instr, score in prev_top_instructions]
        )
        chat_prompt = self._optimizer_prompt_template.fill_template(
            {
                self.EXEMPLARS: self._task_exemplars_fn(),
                self.SCORED_INSTRUCTIONS: formatted_instructions,
                self.NUM_NEW_INSTRUCTIONS: str(num_instructions_to_generate),
            }
        )
        response = self._prompt_optimizer_llm.response(messages=chat_prompt, max_tokens=2048)
        instructions = [Instruction(str_instruction) for str_instruction in self._instruction_extraction_fn(response)]
        return instructions


class OproOptimizer:
    def __init__(
        self,
        instruction_generator: InstructionGenerator,
        train_task: Task,
        validation_task: Task,
        num_train_iterations: int = 5,
        num_new_instructions_per_iter: int = 5,
        num_prev_instructions_to_use: int = 10,
        early_stopping_strategy: EarlyStoppingStrategy | None = None,
    ) -> None:
        self._instruction_generator = instruction_generator
        self._train_task = train_task
        self._validation_task = validation_task
        self._train_iterations = num_train_iterations
        self._num_new_instructions_per_iter = num_new_instructions_per_iter
        self._num_prev_instructions_to_use = num_prev_instructions_to_use
        self._early_stopping_strategy = early_stopping_strategy or RedundantInstructionStopper(
            self._num_new_instructions_per_iter // 2
        )

    def run(self, seed_instructions: list[Instruction]) -> OptimizedInstructions:
        opro_state = _OproState()
        seed_instructions_with_scores = self._score_instructions(list(set(seed_instructions)), task=self._train_task)
        opro_state.add_new_instructions(seed_instructions_with_scores)

        for i in trange(1, self._train_iterations + 1, desc="OPRO: Running prompt optimization"):
            new_instructions = self._instruction_generator.generate_instructions(
                prev_top_instructions=opro_state.top_scored_instructions_with_scores(
                    self._num_prev_instructions_to_use
                ),
                num_instructions_to_generate=self._num_new_instructions_per_iter,
            )
            # dedup instructions
            actual_new_instructions = opro_state.remove_seen_instructions(new_instructions)
            scored_instructions = self._score_instructions(actual_new_instructions, task=self._train_task)
            self._log_train_iteration(scored_instructions, iteration=i)
            opro_state.add_new_instructions(scored_instructions)

            if self._early_stopping_strategy.should_stop(scored_instructions):
                logger.warning(f"Stopping early at iteration {i} because of early stopping criteria.")
                break

        logger.info("OPRO: Running validation scoring on top instructions.")
        validation_scored_instructions = self._score_instructions(
            opro_state.top_scored_instructions(self._num_prev_instructions_to_use),
            task=self._validation_task,
        )
        opro_state.add_validation_scored_instructions(validation_scored_instructions)
        optimized_instructions = opro_state.create_optimized_instructions()
        _best_validation_score = optimized_instructions.best_validation_instruction[1].score
        logger.info(f"OPRO: Best validation score: {_best_validation_score:.4f}")

        return optimized_instructions

    def _log_train_iteration(self, scored_instructions: list[ScoredInstruction], iteration: int) -> None:
        if scored_instructions:
            average_score = sum([score.score for _, score in scored_instructions]) / len(scored_instructions)
            logger.info(
                f"OPRO: Iteration {iteration}: Generated {len(scored_instructions)} new instructions with "
                f"average score {average_score:.4f}"
            )
        else:
            logger.info(f"OPRO: Iteration {iteration}: No new instructions.")

    @staticmethod
    def _score_instructions(instructions: list[Instruction], task: Task) -> list[ScoredInstruction]:
        instruction_with_scores = []
        for instruction in tqdm(instructions, desc="OPRO: Scoring instructions"):
            instruction_with_scores.append((instruction, task.score(instruction)))
        return instruction_with_scores
