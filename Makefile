K_VERSION   ?= $(shell cat deps/k_release)
KOMPILE     ?= $(shell which kompile)

default: help

help:
	@echo "Please read the Makefile."

.PHONY: docker
docker: TAG=runtimeverificationinc/imp-semantics:$(K_VERSION)
docker: package/Dockerfile
	docker build . \
		--build-arg K_VERSION=$(K_VERSION) \
		--file $< \
		--tag $(TAG)

build: build-kimp

build-kimp: have-k
	$(MAKE) -C kimp kdist

.PHONY: have-k
have-k: FOUND_VERSION=$(shell $(KOMPILE) --version | sed -n -e 's/^K version: *v\?\(.*\)$$/\1/p')
have-k:
	@[ ! -z "$(KOMPILE)" ] || \
		(echo "K compiler (kompile) not found (use variable KOMPILE to override)."; exit 1)
	@[ ! -z "$(FOUND_VERSION)" ] || \
		(echo "Unable to determine K compiler ($(KOMPILE)) version."; exit 1)
	@[ "$(K_VERSION)" = "$(FOUND_VERSION)" ] || \
		echo "Unexpected kompile version $(FOUND_VERSION) (expected $(K_VERSION)). Trying anyway..."
