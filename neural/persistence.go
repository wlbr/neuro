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

package neural

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
)

// StorageFormat identifies the file format used to persist a Network.
type StorageFormat string

const (
	// FormatJSON stores the model as indented JSON. The file is human-readable
	// and can be inspected or edited with any text editor.
	FormatJSON StorageFormat = "json"

	// FormatGOB stores the model using Go's encoding/gob format. The file is
	// more compact than JSON but is only readable by Go programs.
	FormatGOB StorageFormat = "gob"
)

// networkSnapshot is the serialisable representation of a Network used for
// JSON and GOB persistence.
type networkSnapshot struct {
	InputNodes          int         `json:"input_nodes"`
	HiddenNodes         int         `json:"hidden_nodes"`
	OutputNodes         int         `json:"output_nodes"`
	LearningRate        float64     `json:"learning_rate"`
	InputHiddenWeights  [][]float64 `json:"input_hidden_weights"`
	HiddenOutputWeights [][]float64 `json:"hidden_output_weights"`
	HiddenBiases        []float64   `json:"hidden_biases"`
	OutputBiases        []float64   `json:"output_biases"`
}

// Save writes the network's architecture and learned parameters to path using
// the given StorageFormat. The file is created (or truncated) at path. Use
// Load to restore the network later.
func (n *Network) Save(path string, format StorageFormat) error {
	snapshot := n.snapshot()

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create model file: %w", err)
	}
	defer file.Close()

	switch format {
	case FormatJSON:
		encoder := json.NewEncoder(file)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(snapshot); err != nil {
			return fmt.Errorf("encode json model: %w", err)
		}
	case FormatGOB:
		if err := gob.NewEncoder(file).Encode(snapshot); err != nil {
			return fmt.Errorf("encode gob model: %w", err)
		}
	default:
		return fmt.Errorf("unsupported storage format %q", format)
	}

	return nil
}

// Load reads a previously saved Network from path using the given
// StorageFormat. The format must match the one used when the file was written.
// The restored Network is ready to use for further training or inference.
func Load(path string, format StorageFormat) (*Network, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open model file: %w", err)
	}
	defer file.Close()

	var snapshot networkSnapshot

	switch format {
	case FormatJSON:
		if err := json.NewDecoder(file).Decode(&snapshot); err != nil {
			return nil, fmt.Errorf("decode json model: %w", err)
		}
	case FormatGOB:
		if err := gob.NewDecoder(file).Decode(&snapshot); err != nil {
			return nil, fmt.Errorf("decode gob model: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported storage format %q", format)
	}

	return networkFromSnapshot(snapshot)
}

// ParseStorageFormat converts a string to a StorageFormat constant. It returns
// an error for any value other than "json" or "gob". Useful for validating
// flag or configuration input before passing it to Save or Load.
func ParseStorageFormat(value string) (StorageFormat, error) {
	format := StorageFormat(value)
	switch format {
	case FormatJSON, FormatGOB:
		return format, nil
	default:
		return "", fmt.Errorf("unsupported storage format %q", value)
	}
}

// snapshot returns a serialisable copy of the network's state.
func (n *Network) snapshot() networkSnapshot {
	return networkSnapshot{
		InputNodes:          n.inputNodes,
		HiddenNodes:         n.hiddenNodes,
		OutputNodes:         n.outputNodes,
		LearningRate:        n.learningRate,
		InputHiddenWeights:  cloneMatrix(n.inputHiddenWeights),
		HiddenOutputWeights: cloneMatrix(n.hiddenOutputWeights),
		HiddenBiases:        append([]float64(nil), n.hiddenBiases...),
		OutputBiases:        append([]float64(nil), n.outputBiases...),
	}
}

// networkFromSnapshot validates and reconstructs a Network from a decoded snapshot.
func networkFromSnapshot(snapshot networkSnapshot) (*Network, error) {
	if snapshot.InputNodes <= 0 || snapshot.HiddenNodes <= 0 || snapshot.OutputNodes <= 0 {
		return nil, fmt.Errorf("snapshot node counts must be positive")
	}
	if snapshot.LearningRate <= 0 {
		return nil, fmt.Errorf("snapshot learning rate must be positive")
	}
	if err := validateMatrix(snapshot.InputHiddenWeights, snapshot.HiddenNodes, snapshot.InputNodes, "input-hidden weights"); err != nil {
		return nil, err
	}
	if err := validateMatrix(snapshot.HiddenOutputWeights, snapshot.OutputNodes, snapshot.HiddenNodes, "hidden-output weights"); err != nil {
		return nil, err
	}
	if len(snapshot.HiddenBiases) != snapshot.HiddenNodes {
		return nil, fmt.Errorf("hidden biases length %d does not match hidden nodes %d", len(snapshot.HiddenBiases), snapshot.HiddenNodes)
	}
	if len(snapshot.OutputBiases) != snapshot.OutputNodes {
		return nil, fmt.Errorf("output biases length %d does not match output nodes %d", len(snapshot.OutputBiases), snapshot.OutputNodes)
	}

	return &Network{
		inputNodes:          snapshot.InputNodes,
		hiddenNodes:         snapshot.HiddenNodes,
		outputNodes:         snapshot.OutputNodes,
		learningRate:        snapshot.LearningRate,
		inputHiddenWeights:  cloneMatrix(snapshot.InputHiddenWeights),
		hiddenOutputWeights: cloneMatrix(snapshot.HiddenOutputWeights),
		hiddenBiases:        append([]float64(nil), snapshot.HiddenBiases...),
		outputBiases:        append([]float64(nil), snapshot.OutputBiases...),
	}, nil
}

// validateMatrix checks that matrix has the expected dimensions.
func validateMatrix(matrix [][]float64, wantRows, wantCols int, name string) error {
	if len(matrix) != wantRows {
		return fmt.Errorf("%s row count %d does not match expected %d", name, len(matrix), wantRows)
	}
	for row := range matrix {
		if len(matrix[row]) != wantCols {
			return fmt.Errorf("%s column count in row %d is %d, expected %d", name, row, len(matrix[row]), wantCols)
		}
	}

	return nil
}

// cloneMatrix returns a deep copy of a two-dimensional slice.
func cloneMatrix(matrix [][]float64) [][]float64 {
	cloned := make([][]float64, len(matrix))
	for row := range matrix {
		cloned[row] = append([]float64(nil), matrix[row]...)
	}

	return cloned
}
