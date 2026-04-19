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

func TestNewNetworkUsesRandomSeed(t *testing.T) {
	n1, err := NewNetwork(2, 3, 1, 0.5)
	if err != nil {
		t.Fatalf("NewNetwork() error = %v", err)
	}

	if n1 == nil {
		t.Fatal("expected non-nil network")
	}

	out1, _ := n1.Query([]float64{0.5, 0.5})

	n2, err := NewNetwork(2, 3, 1, 0.5)
	if err != nil {
		t.Fatalf("NewNetwork() error = %v", err)
	}

	out2, _ := n2.Query([]float64{0.5, 0.5})

	if out1[0] == out2[0] {
		t.Error("NewNetwork with random seeds produced identical outputs (statistically unlikely)")
	}
}

func TestTransposeMatVecMulEdgeCases(t *testing.T) {
	t.Parallel()

	matrix := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	vector := []float64{10, 20}

	result := transposeMatVecMul(matrix, vector)

	expected := []float64{10*1 + 20*4, 10*2 + 20*5, 10*3 + 20*6}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("transposeMatVecMul[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestTransposeMatVecMulSingleRow(t *testing.T) {
	t.Parallel()

	matrix := [][]float64{{1, 2, 3}}
	vector := []float64{5}

	result := transposeMatVecMul(matrix, vector)

	if len(result) != 3 {
		t.Errorf("expected 3 elements, got %d", len(result))
	}

	expected := []float64{5, 10, 15}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("result[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestTransposeMatVecMulPanicsOnMismatch(t *testing.T) {
	t.Parallel()

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on vector length mismatch")
		}
	}()

	matrix := [][]float64{{1, 2}, {3, 4}}
	vector := []float64{1}

	transposeMatVecMul(matrix, vector)
}

func TestAddInPlacePanicsOnMismatch(t *testing.T) {
	t.Parallel()

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()

	dst := []float64{1, 2, 3}
	src := []float64{1, 2}

	addInPlace(dst, src)
}

func TestParseStorageFormatCaseInsensitive(t *testing.T) {
	t.Parallel()

	tests := []string{"json", "gob"}

	for _, tc := range tests {
		f, err := ParseStorageFormat(tc)
		if err != nil {
			t.Errorf("ParseStorageFormat(%q) error = %v", tc, err)
		}
		if f != StorageFormat(tc) {
			t.Errorf("ParseStorageFormat(%q) = %v, want %v", tc, f, tc)
		}
	}
}

func TestLoadWithMissingFile(t *testing.T) {
	t.Parallel()

	if _, err := Load("/nonexistent/path/model.json", FormatJSON); err == nil {
		t.Fatal("expected error loading nonexistent file")
	}
}

func TestSaveAndLoadGOBFormat(t *testing.T) {
	t.Parallel()

	n, _ := NewNetworkWithSeed(2, 3, 1, 0.5, 42)
	path := filepath.Join(t.TempDir(), "model.gob")

	if err := n.Save(path, FormatGOB); err != nil {
		t.Fatalf("Save error = %v", err)
	}

	loaded, err := Load(path, FormatGOB)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}

	if loaded == nil {
		t.Fatal("expected non-nil network")
	}
}

func TestValidateMatrixRejectsWrongDimensions(t *testing.T) {
	t.Parallel()

	matrix := [][]float64{{1, 2}, {3, 4, 5}}

	err := validateMatrix(matrix, 2, 2, "test")
	if err == nil {
		t.Fatal("expected error for inconsistent column counts")
	}
}

func TestNetworkFromSnapshotValidatesWeightDimensions(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 2,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0.5,
		"input_hidden_weights": [[1,2]],
		"hidden_output_weights": [[1,2,3]],
		"hidden_biases": [0,0,0],
		"output_biases": [0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with wrong weight dimensions")
	}
}

func TestNetworkFromSnapshotValidatesBiasDimensions(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 2,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0.5,
		"input_hidden_weights": [[1,2],[3,4],[5,6]],
		"hidden_output_weights": [[1,2,3]],
		"hidden_biases": [0,0],
		"output_biases": [0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with wrong bias dimensions")
	}
}

func TestSaveCreatesValidJSONFile(t *testing.T) {
	t.Parallel()

	n, _ := NewNetworkWithSeed(2, 3, 1, 0.5, 42)
	path := filepath.Join(t.TempDir(), "test.json")

	if err := n.Save(path, FormatJSON); err != nil {
		t.Fatalf("Save error = %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile error = %v", err)
	}

	if len(data) == 0 {
		t.Fatal("saved file is empty")
	}
}

func TestSaveToInvalidPath(t *testing.T) {
	t.Parallel()

	n, _ := NewNetworkWithSeed(2, 3, 1, 0.5, 42)

	if err := n.Save("/invalid/nonexistent/path/model.json", FormatJSON); err == nil {
		t.Fatal("expected error saving to invalid path")
	}
}

func TestLoadInvalidLearningRate(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 2,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0,
		"input_hidden_weights": [[1,2],[3,4],[5,6]],
		"hidden_output_weights": [[1,2,3]],
		"hidden_biases": [0,0,0],
		"output_biases": [0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with invalid learning rate")
	}
}

func TestSaveAndLoadPreservesExactWeights(t *testing.T) {
	t.Parallel()

	n, _ := NewNetworkWithSeed(2, 3, 1, 0.7, 99)

	input := []float64{0.3, 0.7}
	want, _ := n.Query(input)

	path := filepath.Join(t.TempDir(), "exact.json")
	n.Save(path, FormatJSON)

	loaded, _ := Load(path, FormatJSON)
	got, _ := loaded.Query(input)

	if want[0] != got[0] {
		t.Errorf("weights not preserved: want %f, got %f", want[0], got[0])
	}
}

func TestTransposeMatVecMulEmptyMatrix(t *testing.T) {
	t.Parallel()

	result := transposeMatVecMul([][]float64{}, []float64{})
	if result != nil {
		t.Errorf("expected nil for empty matrix, got %v", result)
	}
}

func TestTrainRejectsWrongInputLength(t *testing.T) {
	t.Parallel()

	n, err := NewNetworkWithSeed(2, 3, 1, 0.3, 1)
	if err != nil {
		t.Fatalf("NewNetworkWithSeed() error = %v", err)
	}

	if err := n.Train([]float64{1}, []float64{0}); err == nil {
		t.Fatal("expected input length error")
	}
}

func TestNetworkFromSnapshotValidatesOutputBiasDimensions(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 2,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0.5,
		"input_hidden_weights": [[1,2],[3,4],[5,6]],
		"hidden_output_weights": [[1,2,3]],
		"hidden_biases": [0,0,0],
		"output_biases": [0,0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with wrong output bias dimensions")
	}
}

func TestNetworkFromSnapshotValidatesHiddenOutputWeightRows(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "bad.json")
	badData := `{
		"input_nodes": 2,
		"hidden_nodes": 3,
		"output_nodes": 1,
		"learning_rate": 0.5,
		"input_hidden_weights": [[1,2],[3,4],[5,6]],
		"hidden_output_weights": [[1,2,3],[4,5,6]],
		"hidden_biases": [0,0,0],
		"output_biases": [0]
	}`
	if err := os.WriteFile(path, []byte(badData), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatJSON); err == nil {
		t.Fatal("expected error loading snapshot with wrong hidden-output weight row count")
	}
}

func TestLoadRejectsCorruptedGOBFile(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "corrupt.gob")
	if err := os.WriteFile(path, []byte("not valid gob data"), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}

	if _, err := Load(path, FormatGOB); err == nil {
		t.Fatal("expected error loading corrupted GOB file")
	}
}
