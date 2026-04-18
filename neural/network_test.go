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
