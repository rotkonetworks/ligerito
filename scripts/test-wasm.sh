#!/usr/bin/env bash
# Run the protocol test suites as wasm32 (SIMD128) under Node's WASI.
# Needs: rustup target add wasm32-wasip1; node >= 20.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings $(pwd)/scripts/wasi-runner.mjs"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
# rayon does not build for wasip1, so test without the parallel feature
cargo test --release --target wasm32-wasip1 -p ligerito \
    --no-default-features --features std,prover,hardware-accel \
    --test cross_vectors --test kat --test soundness --test config_consistency \
    -- --test-threads=1 "$@"
