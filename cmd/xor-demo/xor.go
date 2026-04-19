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

import "github.com/wlbr/neuro/neural"

// TrainingSample represents a single XOR training example.
type TrainingSample struct {
	Inputs []float64
	Target []float64
}

// XORData returns the four XOR training samples.
func XORData() []TrainingSample {
	return []TrainingSample{
		{Inputs: []float64{0, 0}, Target: []float64{0}},
		{Inputs: []float64{0, 1}, Target: []float64{1}},
		{Inputs: []float64{1, 0}, Target: []float64{1}},
		{Inputs: []float64{1, 1}, Target: []float64{0}},
	}
}

// TrainXOR trains a network on the XOR problem using the given seed and epochs.
func TrainXOR(epochs int, seed int64) (*neural.Network, error) {
	network, err := neural.NewNetworkWithSeed(2, 4, 1, 0.7, seed)
	if err != nil {
		return nil, err
	}

	data := XORData()
	for epoch := 0; epoch < epochs; epoch++ {
		for _, sample := range data {
			if err := network.Train(sample.Inputs, sample.Target); err != nil {
				return nil, err
			}
		}
	}

	return network, nil
}
