# The K Semantics of IMP

IMP is a toy programming language with mutable variables and sequential execution.

This project showcases the modern methodology of defining language semantics in K.

KIMP consists of two major components:
* The K definition of IMP;
* The `kimp` command-line tool and Python package, that acts as a frontend to the K definitions.


## Trying it out in `docker` (EASY)

The project defines a Docker image that allows both using `kimp` as-is and hacking on it.

First off, clone the project and step into its directory:

```
$ git clone https://github.com/runtimeverification/imp-semantics
$ cd imp-semantics
```

Then, build the image:

```
$ make docker TAG=imp-semantics:latest
```

Run the following command to start a container with an interactive shell:

```
$ docker run --rm -it imp-semantics:latest /bin/bash
```

The `examples` folder, as well as a test script `smoke-tests.sh` is already copied into the workspace.
You can run the tests with:

```
$ ./smoke-test.sh
```

To work with files from the host, run the countainer with a volume mounted.
For example, the following command starts the container and mounts the current working directory under `~/workspace`, ensuring you can work on the examples and have them transparently available in the container.

```
$ docker run --rm -it -v "$PWD":/home/k-user/workspace -u $(id -u):$(id -g) imp-semantics:latest /bin/bash
```

If everything is up and running, feel free to jump straight to the **Usage** section below. If you don't want to use `docker`, read the next section to build `kimp` manually.


## Installation Instructions (ADVANCED)

### Prerequisites

Make sure the K Framework is installed and is available on `PATH`. To install K, follow the official [Quick Start](https://github.com/runtimeverification/k#quick-start) instructions.


### Standalone Installation

`kimp` is a Python package that can be installed using `pip`, ideally into a virtual environment:

```
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install .
```

After installing the package, the K definitions defined by `kimp` have to be kompiled..
To kompile the K definitions for both the LLVM (for concrete execution) and Haskell (for symbolic execution) backend, execute:

```
(venv) $ kdist --verbose build -j2 'imp-semantics.*'
```

After installing the project, you can access the `kimp` CLI:

```
(venv) $ kimp --help
```


### For Developers

Install `poetry` following the instruction [here](https://python-poetry.org/docs/#installation).
Run `poetry install` to transparently install the package into a virtual environment managed by `poetry`.

Use `make` to run common tasks (see the [Makefile](Makefile) for a complete list of available targets).

* `make`: Check code style and run unit tests (also runs `poetry install`)
* `make kdist`: Kompile K definitions
* `make check`: Check code style
* `make test-unit`: Run unit tests
* `make format`: Format code

Command `make kdist` needs to be repeated each time the `.k` files under `kimp/src/kimp/kdist/imp-semantics` change,

Use `poetry run kimp` to execute the `kimp` CLI.
To avoid prefixing every command with `poetry run`, activate the virtual environment with `poetry shell`.


## Usage

`kimp` is intended to demonstrate the two main function of the K framework:
* running example IMP programs using K's concrete execution backend
* proving *claims* about IMP programs by executing them with K's symbolic execution backend

Run `kimp --help` to see the available commands and their arguments. Let us now give examples for both concrete executing and proving:


### Concrete Execution

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


### Symbolic Execution

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
