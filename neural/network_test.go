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
	"os"
	"path/filepath"
	"testing"
)

func TestNewNetworkRejectsInvalidConfiguration(t *testing.T) {
	t.Parallel()

	if _, err := NewNetworkWithSeed(0, 3, 1, 0.3, 1); err == nil {
		t.Fatal("expected invalid node count error")
	}

	if _, err := NewNetworkWithSeed(2, 3, 1, 0, 1); err == nil {
		t.Fatal("expected invalid learning rate error")
	}
}

func TestQueryRejectsWrongInputLength(t *testing.T) {
	t.Parallel()

	n, err := NewNetworkWithSeed(2, 3, 1, 0.3, 1)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	if _, err := n.Query([]float64{1}); err == nil {
		t.Fatal("expected input length error")
	}
}

func TestNetworkLearnsXOR(t *testing.T) {
	n, err := NewNetworkWithSeed(2, 4, 1, 0.7, 1)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	trainingData := []struct {
		inputs []float64
		target []float64
	}{
		{inputs: []float64{0, 0}, target: []float64{0}},
		{inputs: []float64{0, 1}, target: []float64{1}},
		{inputs: []float64{1, 0}, target: []float64{1}},
		{inputs: []float64{1, 1}, target: []float64{0}},
	}

	for epoch := 0; epoch < 15000; epoch++ {
		for _, sample := range trainingData {
			if err := n.Train(sample.inputs, sample.target); err != nil {
				t.Fatalf("Train() error = %v", err)
			}
		}
	}

	assertPrediction := func(inputs []float64, wantMin, wantMax float64) {
		t.Helper()

		outputs, err := n.Query(inputs)
		if err != nil {
			t.Fatalf("Query(%v) error = %v", inputs, err)
		}

		got := outputs[0]
		if got < wantMin || got > wantMax {
			t.Fatalf("Query(%v) = %f, want range [%f, %f]", inputs, got, wantMin, wantMax)
		}
	}

	assertPrediction([]float64{0, 0}, 0.0, 0.2)
	assertPrediction([]float64{0, 1}, 0.8, 1.0)
	assertPrediction([]float64{1, 0}, 0.8, 1.0)
	assertPrediction([]float64{1, 1}, 0.0, 0.2)
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	t.Parallel()

	formats := []struct {
		name   string
		format StorageFormat
		file   string
	}{
		{name: "json", format: FormatJSON, file: "model.json"},
		{name: "gob", format: FormatGOB, file: "model.gob"},
	}

	for _, tc := range formats {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, err := NewNetworkWithSeed(2, 4, 1, 0.7, 1)
			if err != nil {
				t.Fatalf("NewNetworkWithSeed() error = %v", err)
			}

			for epoch := 0; epoch < 5000; epoch++ {
				for _, sample := range []struct {
					inputs []float64
					target []float64
				}{
					{inputs: []float64{0, 0}, target: []float64{0}},
					{inputs: []float64{0, 1}, target: []float64{1}},
					{inputs: []float64{1, 0}, target: []float64{1}},
					{inputs: []float64{1, 1}, target: []float64{0}},
				} {
					if err := n.Train(sample.inputs, sample.target); err != nil {
						t.Fatalf("Train() error = %v", err)
					}
				}
			}

			want, err := n.Query([]float64{0, 1})
			if err != nil {
				t.Fatalf("Query() error = %v", err)
			}

			path := filepath.Join(t.TempDir(), tc.file)
			if err := n.Save(path, tc.format); err != nil {
				t.Fatalf("Save() error = %v", err)
			}

			loaded, err := Load(path, tc.format)
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}

			got, err := loaded.Query([]float64{0, 1})
			if err != nil {
				t.Fatalf("loaded Query() error = %v", err)
			}

			if got[0] != want[0] {
				t.Fatalf("loaded prediction = %f, want %f", got[0], want[0])
			}
		})
	}
}

func TestParseStorageFormatRejectsInvalidValue(t *testing.T) {
	t.Parallel()

	if _, err := ParseStorageFormat("yaml"); err == nil {
		t.Fatal("expected invalid format error")
	}
}

func TestTrainRejectsWrongTargetLength(t *testing.T) {
	t.Parallel()

	n, err := NewNetworkWithSeed(2, 3, 1, 0.3, 1)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	if err := n.Train([]float64{0, 1}, []float64{1, 0}); err == nil {
		t.Fatal("expected target length error")
	}
}

func TestNewNetworkDifferentArchitectures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		inputs, hidden, outputs int
	}{
		{1, 2, 1},
		{3, 5, 2},
		{2, 1, 3},
		{10, 20, 5},
	}

	for _, tc := range tests {
		n, err := NewNetworkWithSeed(tc.inputs, tc.hidden, tc.outputs, 0.5, 42)
		if err != nil {
			t.Fatalf("NewNetworkWithSeed(%d,%d,%d) error = %v", tc.inputs, tc.hidden, tc.outputs, err)
		}

		if n == nil {
			t.Fatal("expected non-nil network")
		}

		outputs, err := n.Query(make([]float64, tc.inputs))
		if err != nil {
			t.Fatalf("Query error = %v", err)
		}

		if len(outputs) != tc.outputs {
			t.Errorf("expected %d outputs, got %d", tc.outputs, len(outputs))
		}
	}
}

func TestOutputsAreInValidRange(t *testing.T) {
	t.Parallel()

	n, err := NewNetworkWithSeed(2, 3, 1, 0.5, 42)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	for i := 0; i < 100; i++ {
		outputs, err := n.Query([]float64{float64(i % 2), float64((i + 1) % 2)})
		if err != nil {
			t.Fatalf("Query error = %v", err)
		}

		for _, out := range outputs {
			if out <= 0 || out >= 1 {
				t.Errorf("output %f out of range (0, 1)", out)
			}
		}
	}
}

func TestDifferentSeedsDifferentWeights(t *testing.T) {
	t.Parallel()

	n1, err := NewNetworkWithSeed(2, 3, 1, 0.5, 1)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	n2, err := NewNetworkWithSeed(2, 3, 1, 0.5, 2)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	out1, _ := n1.Query([]float64{0.5, 0.5})
	out2, _ := n2.Query([]float64{0.5, 0.5})

	if out1[0] == out2[0] {
		t.Error("networks with different seeds produced identical outputs")
	}
}

func TestSameSeedProducesIdenticalBehavior(t *testing.T) {
	t.Parallel()

	n1, _ := NewNetworkWithSeed(2, 4, 1, 0.7, 42)
	n2, _ := NewNetworkWithSeed(2, 4, 1, 0.7, 42)

	input := []float64{0.3, 0.7}
	target := []float64{0.5}

	for i := 0; i < 10; i++ {
		n1.Train(input, target)
		n2.Train(input, target)
	}

	out1, _ := n1.Query(input)
	out2, _ := n2.Query(input)

	if out1[0] != out2[0] {
		t.Errorf("identical networks produced different outputs: %f vs %f", out1[0], out2[0])
	}
}

func TestLoadRejectsCorruptedFile(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "corrupt.json")
	if err := os.WriteFile(path, []byte("{invalid json"), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading corrupted JSON file")
	}
}

func TestLoadRejectsInvalidFormat(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "model.unknown")
	if err := os.WriteFile(path, []byte("{}"), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, StorageFormat("unknown")); err == nil {
		t.Fatal("expected error for unsupported storage format")
	}
}

func TestSaveRejectsInvalidFormat(t *testing.T) {
	t.Parallel()

	n, _ := NewNetworkWithSeed(2, 3, 1, 0.5, 1)
	path := filepath.Join(t.TempDir(), "model.unknown")

	if err := n.Save(path, StorageFormat("unknown")); err == nil {
		t.Fatal("expected error for unsupported storage format")
	}
}

func TestLoadRejectsInvalidNodeCounts(t *testing.T) {
	t.Parallel()

	// Create a valid snapshot but corrupt the node counts
	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 0,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0.5,
		"input_hidden_weights": [],
		"hidden_output_weights": [],
		"hidden_biases": [0,0,0],
		"output_biases": [0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with invalid node count")
	}
}
