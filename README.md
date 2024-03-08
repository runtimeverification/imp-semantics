# The K Semantics of IMP

IMP is a toy programming language with mutable variables and sequential execution.

This project showcases the modern methodology of defining language semantics in K.

KIMP consists of two major components:
* The K definition of IMP;
* The `kimp` command-line tool and Python package, that acts as a frontend to the K definition.

## Trying it out in `docker` (EASY)

We have prepared a docker image that allows both using `kimp` as-is and hacking on it. Use the following to start a container with an interactive shell:

```
docker run --rm -it -v "$PWD":/home/k-user/workspace -u $(id -u):$(id -g) geo2a/bob24:latest /bin/bash
```

This command will download the docker image and mount the current working directory under `~/workspace`. You will have write access to the examples from within the container.

If everything is up and running, feel free to jump straight to the **Usage** section below. If you don't want to use `docker`, read the next section to build `kimp` manually.

## Installation instructions (ADVANCED)

### Prerequisites

Make sure the K Framework is installed and is available on `PATH`. To install K, follow the official [Quick Start](https://github.com/runtimeverification/k#quick-start) instructions.

To build the `kimp` Ptyhon CLI and library, we recommend using the `poetry` Python build tool. Install `poetry` following the instruction [here](https://python-poetry.org/docs/#installation).

### Building

To build the whole codebase, inclusing the `kimp` CLI and the K definition with both backends, LLVM (for concrete execution) and Haskell (for symbolic execution), execute:
```
make build
```

The `kimp` executable is a relatively thin wrapper for a number of generic K tools. These tools need access to the output of the K compiler that were produced at the previous step. The most robust way to point `kimp` to the K compiler output is by setting the following three environment variables:

```
export KIMP_LLVM_DIR=$(realpath ./kimp/kdist/v6.3.25/llvm)
export KIMP_HASKELL_DIR=$(realpath ./kimp/kdist/v6.3.25/haskell)
export KIMP_K_SRC=$(realpath ./kimp/k-src)
```

### Installing

After building the project, you can access the `kimp` CLI via `poetry`:

```
poetry run kimp --help
```

use `poetry shell` to avoid prefixing every command with `poetry run`.

Alternatively, you can install `kimp` with `pip` into a virtual environment:

```
python -m venv venv
source venv/bin/activate
make -C kimp install # this calls pip for you
```

Within that virtual environment, you can use `kimp` directly.


## Usage

`kimp` is intended to demonstrate the two main function of the K framework:
* running example IMP programs using K's concrete execution backend
* proving *claims* about IMP programs by executing them with K's symbolic execution backend

Run `kimp --help` to see the available commands and their arguments. Let us now give examples for both concrete executing and proving:

### Concrete execution

The K Framework generates an LLVM interpreter from the language semantics. Let is see what it does on a simple example program:

```
kimp run examples/sumto10.imp
```

this program adds up the natural numbers up to 10 and should give the following output configuration:

```
<generatedTop>
  <k>
    .K
  </k>
  <env>
    k |-> 11
    n |-> 10
    sum |-> 55
  </env>
</generatedTop>
```

### Symbolic execution

The K Framework is equipped with a symbolic execution backend that can be used to prove properties of programs. The properties to prove are formulated as K claims, and are essentially statements that one state always rewrites to another state if certain conditions hold. An example K claim that formulates an inductive invariant about the summation program we've executed before can be found in `examples/specs/imp-sum-spec.k`. Let us ask the prover to check this claim:

```
kimp prove examples/specs/imp-sum-spec.k IMP-SUM-SPEC sum-spec
```

That command would run for some time and output the symbolic execution trace to a file upon completion. We can pretty-print the trace:

```
kimp show-kcfg IMP-SUM-SPEC sum-spec
```

or even explore it interactively in a terminal user interface

```
kimp view-kcfg IMP-SUM-SPEC sum-spec
```





