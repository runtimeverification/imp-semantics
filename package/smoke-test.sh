#!/usr/bin/env bash

set -euxo pipefail

kimp --help

kimp run --verbose examples/sumto10.imp --env 'x=0,y=1' --env b1=false --env b2=true

kimp prove --verbose examples/specs/imp-sum-spec.k IMP-SUM-SPEC sum-spec

kimp show --verbose IMP-SUM-SPEC sum-spec
