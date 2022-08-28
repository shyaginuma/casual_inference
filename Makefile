.PHONY: lint
lint:
	poetry run black . --check
	poetry run isort . --check
	poetry run mypy .
	poetry run flake8 .

.PHONY: fmt
fmt:
	poetry run black .
	poetry run isort .

.PHONY: test
test:
	poetry run pytest .
