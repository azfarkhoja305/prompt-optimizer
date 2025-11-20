from textwrap import dedent

import pytest

from prompt_optimizer.chat_model import StubChatModel
from prompt_optimizer.chat_prompt import FStringChatTemplate
from prompt_optimizer.opro import (
    RedundantInstructionStopper,
    extract_instructions_between_tags,
    format_scored_instructions,
    _OproState,
    validate_missing_placeholders,
    OproOptimizer,
    LLMInstructionGenerator,
)
from prompt_optimizer.instruction import (
    InstructionScore,
    Instruction,
    OptimizedInstructions,
    Task,
    InstructionGenerator,
)


def _fixed_exemplar_fn() -> str:
    return "Exemplar tasks: Task 1, Task 2"


def test__extract_instructions_between_tags() -> None:
    text = """
    Here are some instructions:
    <instruction>Instruction 1</instruction>
    Some other text.
    <instruction>Instruction 2</instruction>
    End of instructions.
    """
    instructions = extract_instructions_between_tags(text)
    assert instructions == ["Instruction 1", "Instruction 2"]


def test__format_scored_instructions() -> None:
    expected_output = dedent("""
    1.
    Instruction: Instruction 1
    Score: 0.90

    2.
    Instruction: Instruction 2
    Score: 0.75
    """)
    scored_instructions = [("Instruction 1", 0.9), ("Instruction 2", 0.75)]
    formatted = format_scored_instructions(scored_instructions)
    assert formatted == expected_output


def test__redundant_instruction_stopper__should_stop__false() -> None:
    stopper = RedundantInstructionStopper(min_each_iteration=2)

    should_stop = stopper.should_stop([(Instruction("Instruction 1"), InstructionScore(0.9))])
    assert should_stop is True

    # Test case where enough new instructions are provided
    should_stop = stopper.should_stop(
        [
            (Instruction("Instruction 1"), InstructionScore(0.9)),
            (Instruction("Instruction 2"), InstructionScore(0.8)),
            (Instruction("Instruction 3"), InstructionScore(0.7)),
        ]
    )
    assert should_stop is False


def test__redundant_instruction_stopper__should_stop__true() -> None:
    stopper = RedundantInstructionStopper(min_each_iteration=2)

    # Test case where not enough new instructions are provided
    should_stop = stopper.should_stop(
        [
            (Instruction("Instruction 1"), InstructionScore(0.9)),
        ]
    )
    assert should_stop is True


def test__opro_state__add_new_instruction() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [(Instruction("Instruction 1"), InstructionScore(0.2)), (Instruction("Instruction 2"), InstructionScore(0.5))]
    )
    assert opro_state.instructions_at_iteration_n == {0: [Instruction("Instruction 1"), Instruction("Instruction 2")]}
    assert opro_state.scored_instructions == {
        Instruction("Instruction 1"): InstructionScore(0.2),
        Instruction("Instruction 2"): InstructionScore(0.5),
    }
    opro_state.add_new_instructions([(Instruction("Instruction 3"), InstructionScore(0.8))])
    assert opro_state.instructions_at_iteration_n == {
        0: [Instruction("Instruction 1"), Instruction("Instruction 2")],
        1: [Instruction("Instruction 3")],
    }
    assert opro_state.scored_instructions == {
        Instruction("Instruction 1"): InstructionScore(0.2),
        Instruction("Instruction 2"): InstructionScore(0.5),
        Instruction("Instruction 3"): InstructionScore(0.8),
    }


def test__opro_state__seen_instruction() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [(Instruction("Instruction 1"), InstructionScore(0.2)), (Instruction("Instruction 2"), InstructionScore(0.5))]
    )
    assert opro_state.seen_instructions == {Instruction("Instruction 1"), Instruction("Instruction 2")}


def test__opro_state__remove_seen_instruction() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [(Instruction("Instruction 1"), InstructionScore(0.2)), (Instruction("Instruction 2"), InstructionScore(0.5))]
    )
    deduped_instructions = opro_state.remove_seen_instructions(
        [Instruction("Instruction 1"), Instruction("Instruction 3")]
    )
    assert deduped_instructions == [Instruction("Instruction 3")]


def test__opro_state__add_validation_scored_instruction() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [(Instruction("Instruction 1"), InstructionScore(0.2)), (Instruction("Instruction 2"), InstructionScore(0.5))]
    )
    assert len(opro_state.validation_scored_instructions) == 0
    opro_state.add_validation_scored_instructions(
        [(Instruction("Instruction 1"), InstructionScore(0.6)), (Instruction("Instruction 2"), InstructionScore(0.4))]
    )
    assert opro_state.validation_scored_instructions == {
        Instruction("Instruction 1"): InstructionScore(0.6),
        Instruction("Instruction 2"): InstructionScore(0.4),
    }


def test__opro_state__top_scored_instructions_with_scores() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [
            (Instruction("Instruction 1"), InstructionScore(0.2)),
            (Instruction("Instruction 2"), InstructionScore(0.5)),
            (Instruction("Instruction 3"), InstructionScore(0.8)),
        ]
    )
    top_instructions = opro_state.top_scored_instructions_with_scores()
    assert top_instructions == [
        (Instruction("Instruction 3"), InstructionScore(0.8)),
        (Instruction("Instruction 2"), InstructionScore(0.5)),
        (Instruction("Instruction 1"), InstructionScore(0.2)),
    ]
    top_one_instruction = opro_state.top_scored_instructions_with_scores(k=1)
    assert top_one_instruction == [(Instruction("Instruction 3"), InstructionScore(0.8))]


def test__opro_state__top_scored_instructions() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [
            (Instruction("Instruction 1"), InstructionScore(0.2)),
            (Instruction("Instruction 2"), InstructionScore(0.5)),
            (Instruction("Instruction 3"), InstructionScore(0.8)),
        ]
    )
    top_instructions = opro_state.top_scored_instructions()
    assert top_instructions == [
        Instruction("Instruction 3"),
        Instruction("Instruction 2"),
        Instruction("Instruction 1"),
    ]
    top_one_instruction = opro_state.top_scored_instructions(k=1)
    assert top_one_instruction == [Instruction("Instruction 3")]


def test__opro_state__create_optimized_instructions() -> None:
    opro_state = _OproState()
    opro_state.add_new_instructions(
        [
            (Instruction("Instruction 1"), InstructionScore(0.2)),
            (Instruction("Instruction 2"), InstructionScore(0.5)),
        ]
    )
    opro_state.add_validation_scored_instructions(
        [
            (Instruction("Instruction 1"), InstructionScore(0.6)),
        ]
    )
    optimized_instructions = opro_state.create_optimized_instructions()
    assert optimized_instructions == OptimizedInstructions(
        instructions_at_iteration_n={0: [Instruction("Instruction 1"), Instruction("Instruction 2")]},
        scored_instructions={
            Instruction("Instruction 1"): InstructionScore(0.2),
            Instruction("Instruction 2"): InstructionScore(0.5),
        },
        validation_scored_instructions={
            Instruction("Instruction 1"): InstructionScore(0.6),
        },
    )


def test__validate_missing_placeholders() -> None:
    validate_missing_placeholders(required_placeholders={"a", "b"}, found_placeholders={"a", "b", "c"})
    with pytest.raises(ValueError):
        validate_missing_placeholders(required_placeholders={"a", "b", "c"}, found_placeholders={"a", "b"})


def test__llm_instruction_generator__generate_instructions() -> None:
    system_message_str = (
        "Generate {NUM_NEW_INSTRUCTIONS} instructions to solve these example tasks {EXEMPLARS}."
        "Here are the previous instructions with their scores: {SCORED_INSTRUCTIONS}."
    )
    prompt_template = FStringChatTemplate([("system", system_message_str)])
    chat_model = StubChatModel(
        ["<instruction>New instruction 1</instruction>\n <instruction>New instruction 2</instruction>"]
    )
    instruction_generator = LLMInstructionGenerator(
        prompt_optimizer_llm=chat_model,
        optimizer_prompt_template=prompt_template,
        task_exemplars_fn=_fixed_exemplar_fn,
    )
    new_instructions = instruction_generator.generate_instructions(
        prev_top_instructions=[(Instruction("Old instruction 1"), InstructionScore(0.9))],
        num_instructions_to_generate=2,
    )
    assert new_instructions == [Instruction("New instruction 1"), Instruction("New instruction 2")]


def test__opro_optimizer__score__instructions() -> None:
    instruction = [Instruction("Test instruction 1"), Instruction("Test instruction 2")]
    instructions_with_score = OproOptimizer._score_instructions(instruction, _StubTask(score=0.5))
    assert instructions_with_score == [
        (Instruction("Test instruction 1"), InstructionScore(0.5)),
        (Instruction("Test instruction 2"), InstructionScore(0.5)),
    ]


def test__opro_optimizer__run() -> None:
    def _sort_list_in_dict(d: dict[int, list[Instruction]]) -> dict[int, list[Instruction]]:
        return {k: sorted(v, key=lambda x: x.instruction) for k, v in d.items()}

    expected_iteration_dict = {
        0: [Instruction("Seed instruction 1")],
        1: [Instruction("New instruction 1"), Instruction("New instruction 2")],
        2: [Instruction("New instruction 3")],
    }
    opro_optimizer = OproOptimizer(
        instruction_generator=_StubInstructionGenerator(
            instructions=[
                [Instruction("New instruction 1"), Instruction("New instruction 2")],
                [Instruction("New instruction 2"), Instruction("New instruction 3")],
            ]
        ),
        train_task=_StubTask(0.9),
        validation_task=_StubTask(0.5),
        num_train_iterations=2,
        num_new_instructions_per_iter=2,
        num_prev_instructions_to_use=2,
    )

    optimized_instructions = opro_optimizer.run(
        seed_instructions=[Instruction("Seed instruction 1")],
    )

    assert _sort_list_in_dict(optimized_instructions.instructions_at_iteration_n) == _sort_list_in_dict(
        expected_iteration_dict
    )
    assert len(optimized_instructions.scored_instructions) == 4
    assert len(optimized_instructions.validation_scored_instructions) == 2


class _StubTask(Task):
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, instruction: Instruction) -> InstructionScore:
        return InstructionScore(self._score)


class _StubInstructionGenerator(InstructionGenerator):
    def __init__(self, instructions: list[list[Instruction]]) -> None:
        self._instructions = instructions

    def generate_instructions(
        self,
        prev_top_instructions: list[tuple[Instruction, InstructionScore]],
        num_instructions_to_generate: int,
    ) -> list[Instruction]:
        return self._instructions.pop(0)
