// Copyright (c) 2026 Michael Wolber
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Package neural implements a single-hidden-layer feedforward neural network
// trained via backpropagation. All matrix arithmetic is done with plain Go
// slices; the package has no external dependencies.
package neural

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Network is a feedforward neural network with one hidden layer.
// Weights are stored as row-major matrices indexed [destination][source],
// so iterating over the outer slice visits destination nodes and iterating
// over the inner slice visits the incoming connections for that node.
// Biases are stored as plain vectors, one value per destination node.
type Network struct {
	inputNodes   int
	hiddenNodes  int
	outputNodes  int
	learningRate float64

	inputHiddenWeights  [][]float64
	hiddenOutputWeights [][]float64
	hiddenBiases        []float64
	outputBiases        []float64
}

// NewNetwork creates a new Network with randomly initialised weights seeded
// from the current wall-clock time. Use NewNetworkWithSeed for reproducible
// results (e.g. in tests or benchmarks).
func NewNetwork(inputNodes, hiddenNodes, outputNodes int, learningRate float64) (*Network, error) {
	return NewNetworkWithSeed(inputNodes, hiddenNodes, outputNodes, learningRate, time.Now().UnixNano())
}

// NewNetworkWithSeed creates a new Network with weights drawn from
// N(0, 1/√fanIn) using the provided random seed. All node counts must be
// positive and learningRate must be greater than zero.
func NewNetworkWithSeed(inputNodes, hiddenNodes, outputNodes int, learningRate float64, seed int64) (*Network, error) {
	if inputNodes <= 0 || hiddenNodes <= 0 || outputNodes <= 0 {
		return nil, fmt.Errorf("node counts must be positive")
	}
	if learningRate <= 0 {
		return nil, fmt.Errorf("learning rate must be positive")
	}

	rng := rand.New(rand.NewSource(seed))

	return &Network{
		inputNodes:          inputNodes,
		hiddenNodes:         hiddenNodes,
		outputNodes:         outputNodes,
		learningRate:        learningRate,
		inputHiddenWeights:  randomMatrix(rng, hiddenNodes, inputNodes, 1/math.Sqrt(float64(inputNodes))),
		hiddenOutputWeights: randomMatrix(rng, outputNodes, hiddenNodes, 1/math.Sqrt(float64(hiddenNodes))),
		hiddenBiases:        randomVector(rng, hiddenNodes, 1),
		outputBiases:        randomVector(rng, outputNodes, 1),
	}, nil
}

// Train runs one forward pass followed by one backpropagation step for the
// given input/target pair. Call it repeatedly across multiple epochs and
// samples to train the network. inputs must have length inputNodes and targets
// must have length outputNodes.
func (n *Network) Train(inputs, targets []float64) error {
	if len(inputs) != n.inputNodes {
		return fmt.Errorf("expected %d inputs, got %d", n.inputNodes, len(inputs))
	}
	if len(targets) != n.outputNodes {
		return fmt.Errorf("expected %d targets, got %d", n.outputNodes, len(targets))
	}

	hiddenInputs := matVecMul(n.inputHiddenWeights, inputs)
	addInPlace(hiddenInputs, n.hiddenBiases)
	hiddenOutputs := apply(hiddenInputs, sigmoid)

	finalInputs := matVecMul(n.hiddenOutputWeights, hiddenOutputs)
	addInPlace(finalInputs, n.outputBiases)
	finalOutputs := apply(finalInputs, sigmoid)

	outputErrors := make([]float64, n.outputNodes)
	outputGradients := make([]float64, n.outputNodes)
	for i := range outputErrors {
		outputErrors[i] = targets[i] - finalOutputs[i]
		outputGradients[i] = outputErrors[i] * finalOutputs[i] * (1 - finalOutputs[i])
	}

	hiddenErrors := transposeMatVecMul(n.hiddenOutputWeights, outputErrors)
	hiddenGradients := make([]float64, n.hiddenNodes)
	for i := range hiddenGradients {
		hiddenGradients[i] = hiddenErrors[i] * hiddenOutputs[i] * (1 - hiddenOutputs[i])
	}

	for outputNode := range n.hiddenOutputWeights {
		for hiddenNode := range n.hiddenOutputWeights[outputNode] {
			n.hiddenOutputWeights[outputNode][hiddenNode] += n.learningRate * outputGradients[outputNode] * hiddenOutputs[hiddenNode]
		}
		n.outputBiases[outputNode] += n.learningRate * outputGradients[outputNode]
	}

	for hiddenNode := range n.inputHiddenWeights {
		for inputNode := range n.inputHiddenWeights[hiddenNode] {
			n.inputHiddenWeights[hiddenNode][inputNode] += n.learningRate * hiddenGradients[hiddenNode] * inputs[inputNode]
		}
		n.hiddenBiases[hiddenNode] += n.learningRate * hiddenGradients[hiddenNode]
	}

	return nil
}

// Query runs a forward pass and returns the output-layer activations for the
// given inputs. inputs must have length inputNodes. The returned slice has
// length outputNodes and each element is in the range (0, 1) because of the
// sigmoid activation.
func (n *Network) Query(inputs []float64) ([]float64, error) {
	if len(inputs) != n.inputNodes {
		return nil, fmt.Errorf("expected %d inputs, got %d", n.inputNodes, len(inputs))
	}

	hiddenInputs := matVecMul(n.inputHiddenWeights, inputs)
	addInPlace(hiddenInputs, n.hiddenBiases)
	hiddenOutputs := apply(hiddenInputs, sigmoid)

	finalInputs := matVecMul(n.hiddenOutputWeights, hiddenOutputs)
	addInPlace(finalInputs, n.outputBiases)
	return apply(finalInputs, sigmoid), nil
}

// randomMatrix allocates a rows×cols matrix whose entries are drawn from
// N(0, scale²) using rng.
func randomMatrix(rng *rand.Rand, rows, cols int, scale float64) [][]float64 {
	matrix := make([][]float64, rows)
	for row := range matrix {
		matrix[row] = make([]float64, cols)
		for col := range matrix[row] {
			matrix[row][col] = rng.NormFloat64() * scale
		}
	}

	return matrix
}

// randomVector allocates a slice of length size whose entries are drawn from
// N(0, scale²) using rng.
func randomVector(rng *rand.Rand, size int, scale float64) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rng.NormFloat64() * scale
	}

	return vector
}

// matVecMul multiplies matrix (rows×cols) by vector (cols) and returns the
// result (rows). It panics if the inner dimensions do not match.
func matVecMul(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for row := range matrix {
		for col := range matrix[row] {
			result[row] += matrix[row][col] * vector[col]
		}
	}

	return result
}

// transposeMatVecMul multiplies the transpose of matrix (rows×cols) by vector
// (rows) and returns the result (cols). It panics if len(vector) != len(matrix).
func transposeMatVecMul(matrix [][]float64, vector []float64) []float64 {
	if len(matrix) == 0 {
		return nil
	}
	if len(vector) != len(matrix) {
		panic(fmt.Sprintf("transposeMatVecMul: vector length %d does not match matrix row count %d", len(vector), len(matrix)))
	}

	result := make([]float64, len(matrix[0]))
	for row := range matrix {
		for col := range matrix[row] {
			result[col] += matrix[row][col] * vector[row]
		}
	}

	return result
}

// apply maps fn element-wise over values and returns a new slice of the same
// length.
func apply(values []float64, fn func(float64) float64) []float64 {
	result := make([]float64, len(values))
	for i, value := range values {
		result[i] = fn(value)
	}

	return result
}

// addInPlace adds src into dst element-wise. It panics if the slices differ in
// length.
func addInPlace(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("addInPlace: length mismatch dst=%d src=%d", len(dst), len(src)))
	}
	for i := range dst {
		dst[i] += src[i]
	}
}

// sigmoid is the logistic activation function: σ(x) = 1 / (1 + e^−x).
// Its output is always in the open interval (0, 1).
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
