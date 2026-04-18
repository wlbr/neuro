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

// Command xor-demo trains a small neural network on the XOR problem and prints
// the predictions for all four input combinations.
//
// By default the network is trained from scratch. Use -save to persist the
// trained model to disk and -load to skip training and reload a previously
// saved model instead.
//
// Usage:
//
//	xor-demo [-format json|gob] [-epochs N] [-save path] [-load path]
package main

import (
	"flag"
	"fmt"
	"log"

	"ai/neural"
)

func main() {
	formatValue := flag.String("format", "json", "model storage format: json or gob")
	savePath := flag.String("save", "", "path to save the trained model")
	loadPath := flag.String("load", "", "path to load a previously trained model")
	epochs := flag.Int("epochs", 15000, "number of training epochs when creating a new model")
	flag.Parse()

	format, err := neural.ParseStorageFormat(*formatValue)
	if err != nil {
		log.Fatal(err)
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

	var network *neural.Network
	if *loadPath != "" {
		network, err = neural.Load(*loadPath, format)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		network, err = neural.NewNetworkWithSeed(2, 4, 1, 0.7, 1)
		if err != nil {
			log.Fatal(err)
		}

		for epoch := 0; epoch < *epochs; epoch++ {
			for _, sample := range trainingData {
				if err := network.Train(sample.inputs, sample.target); err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	if *savePath != "" {
		if err := network.Save(*savePath, format); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("saved model to %s using %s\n", *savePath, format)
	}

	for _, sample := range trainingData {
		outputs, err := network.Query(sample.inputs)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("inputs=%v target=%v prediction=%.4f\n", sample.inputs, sample.target, outputs[0])
	}
}
