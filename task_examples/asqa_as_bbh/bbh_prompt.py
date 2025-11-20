from textwrap import dedent

META_SYSTEM_PROMPT = dedent("""
Your task is to generate new prompt instructions for a model whose job is to solve a particular user task.

Here are some examples of task the model needs to solve:
{EXEMPLARS}

Here are the best-performing prompt instructions so far, along with their scores.
{SCORED_INSTRUCTIONS}

Generate {NUM_NEW_INSTRUCTIONS} new unique prompt instructions that different from all existing instructions above and 
are expected to achieve a higher score than all previous instructions.

Each instruction should begin <instruction> and ends with </instruction>.
""")


TASK_SOLVER_SYSTEM_PROMPT = "{INSTRUCTION}"
TASK_SOLVER_USER_PROMPT = "{TASK_SAMPLE}"
