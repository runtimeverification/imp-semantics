# Latest versions at the time of writing
K_VERSION   ?= 6.3.19
PYK_VERSION ?= 0.1.664
KOMPILE     ?= $(shell which kompile)

default: help

help:
	@echo "Please read the Makefile."

.phony: docker
docker: docker/.image

docker/.image: docker/Dockerfile.k+pyk
	docker build \
		--build-arg K_VERSION=$(K_VERSION) \
		--build-arg PYK_VERSION=$(PYK_VERSION) \
		-f $< -t runtimeverification/imp-semantics-k:$(K_VERSION) .
	touch $@

K_SOURCES = $(wildcard kimp/k-src/*.k)
TARGETS   = $(patsubst %.k,.build/%-kompiled,$(notdir $(K_SOURCES)))

build: build-llvm build-haskell

build-llvm: have-k $(TARGETS:=-llvm)
build-haskell: have-k $(TARGETS:=-haskell)

.build/%-kompiled-llvm: kimp/k-src/%.k $(K_SOURCES)
	$(KOMPILE) --output-definition $@-llvm $< -I kimp/k-src --backend llvm
.build/%-kompiled-haskell: kimp/k-src/%.k $(K_SOURCES)
	$(KOMPILE) --output-definition $@-haskell $< -I kimp/k-src --backend haskell

.phony: have-k
have-k: FOUND_VERSION = $(shell $(KOMPILE) --version \
				  | sed -n -e 's/^K version: *v\?\([0-9.]*\)$$/\1/p')
have-k:
	@[ ! -z "$(KOMPILE)" ] || \
		(echo "K compiler (kompile) not found (use variable KOMPILE to override)."; exit 1)
	@[ ! -z "$(FOUND_VERSION)" ] || \
		(echo "Unable to determine K compiler ($(KOMPILE)) version."; exit 1)
	@[ "$(K_VERSION)" = "$(FOUND_VERSION)" ] || \
		echo "Unexpected kompile version $(FOUND_VERSION) (expected $(K_VERSION)). Trying anyway..."