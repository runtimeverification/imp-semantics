# KIMP --- K Semantics of IMP

IMP is a toy programming language with mutable variables and sequential execution.

This project showcases the modern methodology of defining language semantics in K.

KIMP consists of two major components:
* The K definition of IMP;
* The `kimp` command-line tool and Python package, that acts as a frontend to the K definition.

## Build instructions

### Prerequisites

Make sure the K Framework is installed and is available on `PATH`. To install K, follow the official [Quick Start](https://github.com/runtimeverification/k#quick-start) instructions.

To build the `kimp` Ptyhon CLI and library, we recommend using the `poetry` Python build tool. Install `poetry` following the instruction [here](https://python-poetry.org/docs/#installation).

### Building

To build the whole codebase, inclusing the `kimp` CLI and the K definition with both backends, LLVM (for concrete execution) and Haskell (for symbolic execution), execute:
```
make build
```

### Accessing compiled K definitions

The definitions are built with `kbuild`, which places the compiled definitions in `~/.kbuild`. To get an exact path to the definition, use:
```
poetry run kbuild which llvm # for LLVM
poetry run kbuild which haskell # for Haskell
```

## Usage

After building the project, you can access the `kimp` CLI via `poetry`:

```
poetry run kimp --help
```
