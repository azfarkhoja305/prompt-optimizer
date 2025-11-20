from prompt_optimizer.instruction import OptimizedInstructions, Instruction, InstructionScore


def test__optimized_instructions__best_validation_instruction() -> None:
    instr1 = OptimizedInstructions(
        instructions_at_iteration_n={0: [Instruction("instr1")], 1: [Instruction("instr2")]},
        scored_instructions={
            Instruction("instr1"): InstructionScore(score=0.9),
            Instruction("instr2"): InstructionScore(score=0.5),
        },
        validation_scored_instructions={
            Instruction("instr1"): InstructionScore(score=0.5),
            Instruction("instr2"): InstructionScore(score=0.8),
        },
    )

    best_instr, best_score = instr1.best_validation_instruction
    assert best_instr == Instruction("instr2")
    assert best_score.score == 0.8
