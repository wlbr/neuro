# neuro 

A feedforward neural network implemented in Go with no external dependencies.

[![Go CI](https://github.com/wlbr/neuro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wlbr/neuro/actions/workflows/ci.yml)
[![Codecov Coverage](https://codecov.io/gh/wlbr/neuro/graph/badge.svg)](https://codecov.io/gh/wlbr/neuro)
[![Go Report Card](https://goreportcard.com/badge/github.com/wlbr/neuro)](https://goreportcard.com/report/github.com/wlbr/neuro)
[![Go Reference](https://pkg.go.dev/badge/github.com/wlbr/neuro.svg)](https://pkg.go.dev/github.com/wlbr/neuro)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The network has one hidden layer, uses sigmoid activation, and trains via backpropagation with bias terms. Weights are initialised using Xavier-like scaling (`N(0, 1/√fanIn)`).

## Requirements

Go 1.25 or later.

## Quick start

```sh
# train a model and print predictions
make demo

# train a tic-tac-toe neural player and evaluate vs Minimax
make tictactoe

# save / reload tic-tac-toe model
make tictactoe-save
make tictactoe-load

# train, save to JSON, then reload and predict
make demo-save
make demo-load

# train, save to GOB, then reload and predict
make demo-save-gob
make demo-load-gob
```

## Using the `neural` package

```go
import "github.com/wlbr/neuro/neural"

// create a network: 2 inputs, 4 hidden nodes, 1 output, learning rate 0.7
net, err := neural.NewNetwork(2, 4, 1, 0.7)

// use NewNetworkWithSeed for reproducible results
net, err := neural.NewNetworkWithSeed(2, 4, 1, 0.7, 42)

// train one sample (call repeatedly over many epochs)
err = net.Train([]float64{0, 1}, []float64{1})

// predict
outputs, err := net.Query([]float64{0, 1})

// save
err = net.Save("model.json", neural.FormatJSON)
err = net.Save("model.gob",  neural.FormatGOB)

// load
net, err = neural.Load("model.json", neural.FormatJSON)
net, err = neural.Load("model.gob",  neural.FormatGOB)
```

## `cmd/xor-demo` flags

| Flag | Default | Description |
|------|---------|-------------|
| `-format` | `json` | Storage format: `json` or `gob` |
| `-save` | _(none)_ | Path to save the trained model |
| `-load` | _(none)_ | Path to load a model (skips training) |
| `-epochs` | `15000` | Training epochs when creating a new model |

## `cmd/tictactoe-demo`

Trains a neural network to play tic-tac-toe by learning from minimax-optimal moves, then evaluates it against both a perfect (Minimax) and a random player.

| Flag | Default | Description |
|------|---------|-------------|
| `-epochs` | `200` | Training epochs |
| `-games` | `10000` | Random games used to generate training data |
| `-hidden` | `36` | Hidden layer nodes |
| `-lr` | `0.1` | Learning rate |
| `-seed` | `42` | Random seed |
| `-eval` | `1000` | Games per evaluation matchup |
| `-format` | `json` | Storage format: `json` or `gob` |
| `-save` | _(none)_ | Path to save the trained model |
| `-load` | _(none)_ | Path to load a model (skips training) |

## Format comparison

| | JSON | GOB |
|-|------|-----|
| Human-readable | ✓ | ✗ |
| Inspectable with a text editor | ✓ | ✗ |
| More compact on disk | ✗ | ✓ |
| Go-to-Go only | ✗ | ✓ |

## Development

```sh
make demo              # run the XOR demo
make tictactoe        # train and evaluate tic-tac-toe neural player
make tictactoe-save   # train and save tic-tac-toe model
make tictactoe-load   # reload and evaluate saved tic-tac-toe model
make test             # run all tests
make test-verbose     # run all tests with -v
make test-race        # run tests with race detection
make test-coverage    # run tests with coverage report
make fmt              # gofmt all Go files
make vet              # go vet all packages
make license-check    # check all files have MIT license headers
make license-add      # add MIT license headers to source files
make check            # fmt + vet + license-check + test-race
make ci               # full CI pipeline (fmt + vet + license-check + test-race + coverage)
make clean            # remove saved model files
```

## Code Quality & Documentation

| Resource | Link |
|----------|------|
| **Package Docs** | [pkg.go.dev](https://pkg.go.dev/github.com/wlbr/neuro) |
| **CI/CD Pipeline** | [GitHub Actions](https://github.com/wlbr/neuro/actions) |
| **Test Coverage** | [Codecov](https://codecov.io/gh/wlbr/neuro) |
| **Code Metrics** | [Go Report Card](https://goreportcard.com/report/github.com/wlbr/neuro) |
| **Lint Results** | [Go Vet](https://github.com/wlbr/neuro/actions?query=workflow%3A%22Go+CI%22) |

All code is automatically formatted with `gofmt`, linted with `go vet`, and tested with race detection enabled.
