# The K Semantics of IMP

IMP is a toy programming language with mutable variables and sequential execution.

This project showcases the modern methodology of defining language semantics in K.

KIMP consists of two major components:
* The K definition of IMP;
* The `kimp` command-line tool and Python package, that acts as a frontend to the K definition.


## Trying it out in `docker` (EASY)

The project defines a Docker image that allows both using `kimp` as-is and hacking on it.

First off, clone the project and step into its directory:

```
git clone https://github.com/runtimeverification/imp-semantics
cd imp-semantics
```

Then, build the image:

```
make docker TAG=imp-semantics:latest
```

Run the following command to start a container with an interactive shell:

```
docker run --rm -it imp-semantics:latest /bin/bash
```

The `examples` folder, as well as a test script `smoke-tests.sh` is already copied into the workspace.
You can run the tests with:

```
./smoke-test.sh
```

To work with files from the host, run the countainer with a volume mounted.
For example, the following command starts the container and mounts the current working directory under `~/workspace`, ensuring you can work on the examples and have them transparently available in the container.

```
docker run --rm -it -v "$PWD":/home/k-user/workspace -u $(id -u):$(id -g) imp-semantics:latest /bin/bash
```

If everything is up and running, feel free to jump straight to the **Usage** section below. If you don't want to use `docker`, read the next section to build `kimp` manually.


## Installation instructions (ADVANCED)

### Prerequisites

Make sure the K Framework is installed and is available on `PATH`. To install K, follow the official [Quick Start](https://github.com/runtimeverification/k#quick-start) instructions.

To build the `kimp` Python CLI and library, we recommend using the `poetry` Python build tool. Install `poetry` following the instruction [here](https://python-poetry.org/docs/#installation).


### Building

To build the whole codebase, inclusing the `kimp` CLI and the K definition with both backends, LLVM (for concrete execution) and Haskell (for symbolic execution), execute:

```
make build
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
pip install kimp
```

Within that virtual environment, you can use `kimp` directly.


## Usage

`kimp` is intended to demonstrate the two main function of the K framework:
* running example IMP programs using K's concrete execution backend
* proving *claims* about IMP programs by executing them with K's symbolic execution backend

Run `kimp --help` to see the available commands and their arguments. Let us now give examples for both concrete executing and proving:


### Preparation

The K files need to be compiled before anything can be executed.
The `Makefile` defines a `build` target that will executed the `kompile` commands for this.

```
make build
```

If the `*.k` files in `kimp/src/kimp/kdist/imp-semantics` change, this step needs to be repeated.


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
kimp show IMP-SUM-SPEC sum-spec
```

or even explore it interactively in a terminal user interface

```
kimp view IMP-SUM-SPEC sum-spec
```
