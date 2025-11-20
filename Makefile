.PHONY: lock install format test-static test-unit

TESTS := tests
SRC := prompt_optimizer
EXAMPLES := task_examples
ALL ?= ${SRC} ${TESTS} ${EXAMPLES}

lock:
	@echo "Locking dependencies..."
	poetry lock --no-update

install:
	@echo "Installing environment..."
	poetry install --all-extras

format:
	poetry run ruff format
	poetry run ruff check --fix

test-static:
	poetry run ruff check
	poetry run mypy ${ALL}

test-unit:
	poetry run pytest tests/unit

test-integration:
	poetry run pytest tests/integration
