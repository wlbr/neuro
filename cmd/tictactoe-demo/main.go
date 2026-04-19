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

// Command tictactoe-demo trains a neural network to play tic-tac-toe by
// learning from minimax-optimal moves. After training it evaluates the
// network against both a perfect (Minimax) and a random player, then
// displays a sample game.
//
// Usage:
//
//	tictactoe-demo [-epochs N] [-games N] [-hidden N] [-lr F] [-seed N]
//	               [-eval N] [-format json|gob] [-save path] [-load path]
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/wlbr/neuro/neural"
)

func main() {
	epochs := flag.Int("epochs", 200, "training epochs")
	games := flag.Int("games", 10000, "random games for training data generation")
	hidden := flag.Int("hidden", 36, "hidden layer nodes")
	lr := flag.Float64("lr", 0.1, "learning rate")
	seed := flag.Int64("seed", 42, "random seed")
	savePath := flag.String("save", "", "save trained model to file")
	loadPath := flag.String("load", "", "load model from file")
	format := flag.String("format", "json", "model format: json or gob")
	evalGames := flag.Int("eval", 1000, "games per evaluation matchup")
	flag.Parse()

	sf, err := neural.ParseStorageFormat(*format)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid format: %v\n", err)
		os.Exit(1)
	}

	var network *neural.Network

	if *loadPath != "" {
		fmt.Printf("Loading model from %s...\n", *loadPath)
		network, err = neural.Load(*loadPath, sf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("  Model loaded.")
	} else {
		fmt.Printf("Generating training data from %d random games...\n", *games)
		samples := GenerateTrainingData(*games, *seed)
		fmt.Printf("  Collected %d unique positions\n\n", len(samples))

		fmt.Printf("Training neural network (9→%d→9, lr=%.2f, seed=%d)...\n",
			*hidden, *lr, *seed)
		network, err = TrainNetwork(samples, *hidden, *epochs, *lr, *seed)
		if err != nil {
			fmt.Fprintf(os.Stderr, "training error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println()
	}

	fmt.Printf("Evaluation (%d games each, alternating sides):\n", *evalGames)

	mm := MinimaxPlayer{}
	mmResult := Evaluate(network, mm, *evalGames, *seed+1)
	fmt.Printf("  vs Minimax:  W=%-4d D=%-4d L=%-4d (draw rate %.1f%%)\n",
		mmResult.Wins, mmResult.Draws, mmResult.Losses,
		100*float64(mmResult.Draws)/float64(*evalGames))

	rp := NewRandomPlayer(*seed + 2)
	rpResult := Evaluate(network, rp, *evalGames, *seed+3)
	fmt.Printf("  vs Random:   W=%-4d D=%-4d L=%-4d (win rate %.1f%%)\n",
		rpResult.Wins, rpResult.Draws, rpResult.Losses,
		100*float64(rpResult.Wins)/float64(*evalGames))

	fmt.Println()
	fmt.Println("Sample game (Neural X vs Minimax O):")
	np := &NeuralPlayer{network: network}
	ShowSampleGame(np, mm)

	if *savePath != "" {
		fmt.Printf("\nSaving model to %s...\n", *savePath)
		if err := network.Save(*savePath, sf); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("  Model saved.")
	}
}
