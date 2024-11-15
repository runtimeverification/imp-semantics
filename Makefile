POETRY     := poetry
POETRY_RUN := $(POETRY) run


default: check test-unit

all: check cov

.PHONY: clean
clean:
	rm -rf dist .coverage cov-* .mypy_cache .pytest_cache
	find -type d -name __pycache__ -prune -exec rm -rf {} \;

.PHONY: build
build:
	$(POETRY) build

.PHONY: poetry-install
poetry-install:
	$(POETRY) install


# Docker

K_VERSION ?= $(shell cat deps/k_release)

.PHONY: docker
docker: TAG=runtimeverificationinc/imp-semantics:$(K_VERSION)
docker: package/Dockerfile
	docker build . \
		--build-arg K_VERSION=$(K_VERSION) \
		--file $< \
		--tag $(TAG)


# Kompilation

KOMPILE ?= $(shell which kompile)

.PHONY: have-k
have-k: FOUND_VERSION=$(shell $(KOMPILE) --version | sed -n -e 's/^K version: *v\?\(.*\)$$/\1/p')
have-k:
	@[ ! -z "$(KOMPILE)" ] || \
		(echo "K compiler (kompile) not found (use variable KOMPILE to override)."; exit 1)
	@[ ! -z "$(FOUND_VERSION)" ] || \
		(echo "Unable to determine K compiler ($(KOMPILE)) version."; exit 1)
	@[ "$(K_VERSION)" = "$(FOUND_VERSION)" ] || \
		echo "Unexpected kompile version $(FOUND_VERSION) (expected $(K_VERSION)). Trying anyway..."

kdist: kdist-build

KDIST_ARGS :=

kdist-build: poetry-install have-k
	$(POETRY_RUN) kdist --verbose build -j2 $(KDIST_ARGS)

kdist-clean: poetry-install
	$(POETRY_RUN) kdist clean


# Tests

TEST_ARGS :=

test: test-all

test-all: poetry-install
	$(POETRY_RUN) pytest src/tests --maxfail=1 --verbose --durations=0 --numprocesses=4 --dist=worksteal $(TEST_ARGS)

test-unit: poetry-install
	$(POETRY_RUN) pytest src/tests/unit --maxfail=1 --verbose $(TEST_ARGS)

test-integration: poetry-install
	$(POETRY_RUN) pytest src/tests/integration --maxfail=1 --verbose --durations=0 --numprocesses=4 --dist=worksteal $(TEST_ARGS)


# Coverage

COV_ARGS :=

cov: cov-all

cov-%: TEST_ARGS += --cov=kimp --no-cov-on-fail --cov-branch --cov-report=term

cov-all: TEST_ARGS += --cov-report=html:cov-all-html $(COV_ARGS)
cov-all: test-all

cov-unit: TEST_ARGS += --cov-report=html:cov-unit-html $(COV_ARGS)
cov-unit: test-unit

cov-integration: TEST_ARGS += --cov-report=html:cov-integration-html $(COV_ARGS)
cov-integration: test-integration


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
