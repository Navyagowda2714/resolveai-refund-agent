.PHONY: install test lint format demo eval serve docker clean

PY ?= python3
export LLM_PROVIDER ?= offline

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

demo:
	resolveai demo

eval:
	resolveai eval --out-dir evals

serve:
	resolveai seed
	resolveai serve --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info data/*.db data/*.json
