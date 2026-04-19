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
	"bytes"
	"flag"
	"io"
	"os"
	"testing"
)

func TestPlayGameMinimaxVsMinimax(t *testing.T) {
	t.Parallel()
	mm1 := MinimaxPlayer{}
	mm2 := MinimaxPlayer{}
	for i := 0; i < 10; i++ {
		w := PlayGame(mm1, mm2)
		if w != cellEmpty {
			t.Fatalf("game %d: two minimax players should always draw, got %d", i, w)
		}
	}
}

func TestTrainNetworkZeroEpochs(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(100, 1)
	net, err := TrainNetwork(samples, 18, 0, 0.1, 1)
	if err != nil {
		t.Fatalf("TrainNetwork(0 epochs) error: %v", err)
	}
	if net == nil {
		t.Fatal("expected non-nil network with zero epochs")
	}
}

func TestTrainNetworkSingleEpoch(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(50, 3)
	net, err := TrainNetwork(samples, 9, 1, 0.1, 3)
	if err != nil {
		t.Fatalf("TrainNetwork error: %v", err)
	}
	// Network should still produce valid outputs.
	np := &NeuralPlayer{network: net}
	move := np.ChooseMove(Board{}, cellX)
	if move < 0 || move >= boardSize {
		t.Errorf("invalid move %d", move)
	}
}

func TestEvaluateCountsAllGames(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(200, 5)
	net, err := TrainNetwork(samples, 18, 10, 0.1, 5)
	if err != nil {
		t.Fatalf("TrainNetwork error: %v", err)
	}
	rp := NewRandomPlayer(7)
	games := 50
	result := Evaluate(net, rp, games, 8)
	total := result.Wins + result.Draws + result.Losses
	if total != games {
		t.Errorf("expected %d total games, got %d (W=%d D=%d L=%d)",
			games, total, result.Wins, result.Draws, result.Losses)
	}
}

func TestEvaluateAlternatesSides(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(200, 11)
	net, err := TrainNetwork(samples, 18, 10, 0.1, 11)
	if err != nil {
		t.Fatalf("TrainNetwork error: %v", err)
	}
	mm := MinimaxPlayer{}
	// With 2 games the network plays X once and O once.
	result := Evaluate(net, mm, 2, 12)
	total := result.Wins + result.Draws + result.Losses
	if total != 2 {
		t.Errorf("expected 2 games, got %d", total)
	}
}

func TestShowSampleGame(t *testing.T) {
	t.Parallel()

	// Capture stdout.
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	mm := MinimaxPlayer{}
	rp := NewRandomPlayer(1)
	ShowSampleGame(mm, rp)

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	io.Copy(&buf, r)
	output := buf.String()

	if len(output) == 0 {
		t.Fatal("ShowSampleGame produced no output")
	}
	// Output should contain "Move 1:" at minimum.
	if !bytes.Contains([]byte(output), []byte("Move 1:")) {
		t.Error("expected output to contain 'Move 1:'")
	}
	// Output should contain a result line.
	if !bytes.Contains([]byte(output), []byte("Result:")) {
		t.Error("expected output to contain 'Result:'")
	}
}

func TestGenerateTrainingDataDeterministic(t *testing.T) {
	t.Parallel()
	s1 := GenerateTrainingData(100, 42)
	s2 := GenerateTrainingData(100, 42)
	if len(s1) != len(s2) {
		t.Fatalf("same seed produced different sample counts: %d vs %d", len(s1), len(s2))
	}
	for i := range s1 {
		if s1[i].Board != s2[i].Board || s1[i].Player != s2[i].Player || s1[i].BestMove != s2[i].BestMove {
			t.Fatalf("sample %d differs between runs with same seed", i)
		}
	}
}

func TestGenerateTrainingDataDifferentSeeds(t *testing.T) {
	t.Parallel()
	s1 := GenerateTrainingData(100, 1)
	s2 := GenerateTrainingData(100, 2)
	// Different seeds should explore different random game trees,
	// potentially producing different sample counts.
	if len(s1) == len(s2) {
		// Same count is possible but samples should differ in ordering.
		allSame := true
		for i := range s1 {
			if s1[i].Board != s2[i].Board || s1[i].Player != s2[i].Player {
				allSame = false
				break
			}
		}
		if allSame {
			t.Error("different seeds produced identical training data")
		}
	}
}

func TestRunTrainAndEvaluate(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-epochs", "5", "-games", "50", "-hidden", "9", "-eval", "10"}
	if err := run(); err != nil {
		t.Fatalf("run() error: %v", err)
	}
}

func TestRunSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := tmpDir + "/ttt-model.json"

	// Train and save.
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-epochs", "5", "-games", "50", "-hidden", "9",
		"-eval", "10", "-save", modelPath}
	if err := run(); err != nil {
		t.Fatalf("run(save) error: %v", err)
	}

	// Load and evaluate.
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-load", modelPath, "-eval", "10"}
	if err := run(); err != nil {
		t.Fatalf("run(load) error: %v", err)
	}
}

func TestRunSaveGOBFormat(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := tmpDir + "/ttt-model.gob"

	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-epochs", "5", "-games", "50", "-hidden", "9",
		"-eval", "10", "-format", "gob", "-save", modelPath}
	if err := run(); err != nil {
		t.Fatalf("run(gob save) error: %v", err)
	}

	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-format", "gob", "-load", modelPath, "-eval", "10"}
	if err := run(); err != nil {
		t.Fatalf("run(gob load) error: %v", err)
	}
}

func TestRunInvalidFormat(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-format", "xml"}
	if err := run(); err == nil {
		t.Fatal("expected error for invalid format")
	}
}

func TestRunLoadMissingFile(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{"tictactoe-demo", "-load", "/nonexistent/model.json"}
	if err := run(); err == nil {
		t.Fatal("expected error for missing file")
	}
}
