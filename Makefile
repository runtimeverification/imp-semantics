# Latest versions at the time of writing
K_VERSION   ?= 6.3.19
PYK_VERSION ?= 0.1.664

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
