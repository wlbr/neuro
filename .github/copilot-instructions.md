# Copilot Instructions

## Build, test, and lint commands

- Run the full test suite with `go test ./...`.
- Run the neural-network tests only with `go test ./neural`.
- Run a single test with `go test ./neural -run TestNetworkLearnsXOR`.
- Run the demo program with `go run ./cmd/xor-demo`.
- Save a model with `go run ./cmd/xor-demo -format json -save xor-model.json` or `go run ./cmd/xor-demo -format gob -save xor-model.gob`.
- Reload a saved model with `go run ./cmd/xor-demo -format json -load xor-model.json` or `go run ./cmd/xor-demo -format gob -load xor-model.gob`.
- Format code with `gofmt -w .`.

## High-level architecture

- `neural` contains the reusable feedforward neural-network implementation. It owns weight initialization, forward passes, and backpropagation for a single hidden-layer network.
- `neural` also owns model persistence, with format-switched save/load support for JSON and GOB snapshots.
- `cmd/xor-demo` is a runnable example that trains the network on the XOR problem and prints predictions. Use it as a usage reference rather than placing application logic in the package itself.
- The repository is intentionally dependency-light: matrix math is implemented with slices and loops instead of external numerical libraries.

## Key conventions

- Keep reusable learning logic in `neural`; put runnable experiments and demos under `cmd/...`.
- Preserve deterministic tests and examples by using `NewNetworkWithSeed` when behavior needs to be reproducible.
- Prefer the shared `Save`, `Load`, and `ParseStorageFormat` helpers instead of adding format-specific persistence code in callers.
- Weight matrices are stored as `[][fanIn]` per destination layer node, so row-major loops should follow `destination -> source`.
- After every change, verify the program still runs: `make demo` must succeed before considering a task done.
- Keep `README.md` and `Makefile` in sync with any new flags, commands, packages, or behaviours introduced. Do not leave them stale.
