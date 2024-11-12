#!/usr/bin/env bash

set -euxo pipefail

kimp --help

kimp run --verbose --input-file examples/sumto10.imp

kimp prove --verbose examples/specs/imp-sum-spec.k IMP-SUM-SPEC sum-spec

kimp show --verbose IMP-SUM-SPEC sum-spec
