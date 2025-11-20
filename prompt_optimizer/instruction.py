from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Protocol
import pickle


@dataclass(frozen=True)
class Instruction:
    instruction: str


@dataclass(frozen=True)
class InstructionScore:
    score: float


@dataclass(frozen=True)
class OptimizedInstructions:
    instructions_at_iteration_n: dict[int, list[Instruction]]
    scored_instructions: dict[Instruction, InstructionScore]
    validation_scored_instructions: dict[Instruction, InstructionScore]

    @property
    def best_validation_instruction(self) -> tuple[Instruction, InstructionScore]:
        best_instruction = max(
            self.validation_scored_instructions.items(),
            key=lambda item: item[1].score,
        )
        return best_instruction

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


class InstructionGenerator(ABC):
    @abstractmethod
    def generate_instructions(
        self, prev_top_instructions: list[tuple[Instruction, InstructionScore]], num_instructions_to_generate: int
    ) -> list[Instruction]:
        pass


class Task(ABC):
    @abstractmethod
    def score(self, instruction: Instruction) -> InstructionScore:
        pass


class EarlyStoppingStrategy(Protocol):
    def should_stop(self, new_instruction_with_scores: list[tuple[Instruction, InstructionScore]]) -> bool:
        pass
