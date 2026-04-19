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

package main

import (
	"testing"
)

func TestXORData(t *testing.T) {
	data := XORData()

	if got := len(data); got != 4 {
		t.Errorf("expected 4 samples, got %d", got)
	}

	tests := []struct {
		inputs []float64
		target float64
	}{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	for i, tt := range tests {
		if len(data[i].Inputs) != 2 {
			t.Errorf("sample %d: expected 2 inputs, got %d", i, len(data[i].Inputs))
		}
		if data[i].Inputs[0] != tt.inputs[0] || data[i].Inputs[1] != tt.inputs[1] {
			t.Errorf("sample %d: expected inputs %v, got %v", i, tt.inputs, data[i].Inputs)
		}
		if data[i].Target[0] != tt.target {
			t.Errorf("sample %d: expected target %v, got %v", i, tt.target, data[i].Target[0])
		}
	}
}

func TestTrainXOR(t *testing.T) {
	network, err := TrainXOR(100, 42)
	if err != nil {
		t.Fatalf("TrainXOR failed: %v", err)
	}

	if network == nil {
		t.Error("expected non-nil network")
	}

	data := XORData()
	for _, sample := range data {
		outputs, err := network.Query(sample.Inputs)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if len(outputs) != 1 {
			t.Errorf("expected 1 output, got %d", len(outputs))
		}

		if outputs[0] < 0 || outputs[0] > 1 {
			t.Errorf("output out of bounds [0,1]: %f", outputs[0])
		}
	}
}

func TestTrainXORProducesReasonablePredictions(t *testing.T) {
	network, err := TrainXOR(5000, 42)
	if err != nil {
		t.Fatalf("TrainXOR failed: %v", err)
	}

	data := XORData()
	for _, sample := range data {
		outputs, err := network.Query(sample.Inputs)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		expected := sample.Target[0]
		prediction := outputs[0]

		if expected > 0.5 && prediction <= 0.5 {
			t.Errorf("inputs %v: expected > 0.5, got %.4f", sample.Inputs, prediction)
		}
		if expected < 0.5 && prediction >= 0.5 {
			t.Errorf("inputs %v: expected < 0.5, got %.4f", sample.Inputs, prediction)
		}
	}
}
