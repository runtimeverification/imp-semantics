POETRY     := poetry
POETRY_RUN := $(POETRY) run


.PHONY: clean
clean:
	rm -rf dist .mypy_cache .pytest_cache
	find -type d -name __pycache__ -prune -exec rm -rf {} \;

.PHONY: build
build: kbuild-imp

.PHONY: poetry-install
poetry-install:
	$(POETRY) install

.PHONY: kbuild-imp
kbuild-imp: poetry-install
	$(POETRY) run kbuild kompile llvm
	$(POETRY) run kbuild kompile haskell

.PHONY: install
install: poetry-install
	$(POETRY) build
	pip install dist/*.whl

# Checks and formatting

format: autoflake isort black
check: check-flake8 check-mypy check-autoflake check-isort check-black

check-flake8: poetry-install
	$(POETRY_RUN) flake8 src

check-mypy: poetry-install
	$(POETRY_RUN) mypy src

autoflake: poetry-install
	$(POETRY_RUN) autoflake --quiet --in-place src

check-autoflake: poetry-install
	$(POETRY_RUN) autoflake --quiet --check src

isort: poetry-install
	$(POETRY_RUN) isort src

check-isort: poetry-install
	$(POETRY_RUN) isort --check src

black: poetry-install
	$(POETRY_RUN) black src

check-black: poetry-install
	$(POETRY_RUN) black --check src


# Optional tools

SRC_FILES := $(shell find src -type f -name '*.py')

pyupgrade: poetry-install
	$(POETRY_RUN) pyupgrade --py310-plus $(SRC_FILES)
