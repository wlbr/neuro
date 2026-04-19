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

import "testing"

func TestBoardWinner(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		board Board
		want  int
	}{
		{"empty", Board{}, cellEmpty},
		{"X row", Board{1, 1, 1, 0, 0, 0, 0, 0, 0}, cellX},
		{"O col", Board{-1, 0, 0, -1, 0, 0, -1, 0, 0}, cellO},
		{"X diag", Board{1, 0, 0, 0, 1, 0, 0, 0, 1}, cellX},
		{"O anti-diag", Board{0, 0, -1, 0, -1, 0, -1, 0, 0}, cellO},
		{"no winner", Board{1, -1, 1, 1, -1, -1, -1, 1, 1}, cellEmpty},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.board.winner(); got != tt.want {
				t.Errorf("winner() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestBoardEmptyCells(t *testing.T) {
	t.Parallel()
	b := Board{1, 0, -1, 0, 1, 0, -1, 0, 1}
	got := b.emptyCells()
	want := []int{1, 3, 5, 7}
	if len(got) != len(want) {
		t.Fatalf("emptyCells() len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("emptyCells()[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestBoardIsOver(t *testing.T) {
	t.Parallel()
	if (Board{}).isOver() {
		t.Error("empty board should not be over")
	}
	full := Board{1, -1, 1, 1, -1, -1, -1, 1, 1}
	if !full.isOver() {
		t.Error("full draw board should be over")
	}
	won := Board{1, 1, 1, 0, 0, 0, 0, 0, 0}
	if !won.isOver() {
		t.Error("won board should be over")
	}
}

func TestBoardEncode(t *testing.T) {
	t.Parallel()
	b := Board{cellX, cellEmpty, cellO, cellEmpty, cellX, cellEmpty, cellO, cellEmpty, cellX}
	got := b.encode(cellX)
	want := []float64{1, 0, -1, 0, 1, 0, -1, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("encode(X)[%d] = %f, want %f", i, got[i], want[i])
		}
	}
	// From O's perspective, values should flip.
	got2 := b.encode(cellO)
	want2 := []float64{-1, 0, 1, 0, -1, 0, 1, 0, -1}
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Errorf("encode(O)[%d] = %f, want %f", i, got2[i], want2[i])
		}
	}
}

func TestBoardString(t *testing.T) {
	t.Parallel()
	b := Board{cellX, cellO, cellEmpty, cellEmpty, cellX, cellEmpty, cellEmpty, cellEmpty, cellO}
	s := b.String()
	if len(s) == 0 {
		t.Fatal("String() returned empty")
	}
}

func TestMinimaxNeverLoses(t *testing.T) {
	t.Parallel()
	mm := MinimaxPlayer{}
	rp := NewRandomPlayer(99)

	for i := 0; i < 200; i++ {
		var w int
		if i%2 == 0 {
			w = PlayGame(mm, rp)
		} else {
			w = PlayGame(rp, mm)
		}
		mmCell := cellX
		if i%2 != 0 {
			mmCell = cellO
		}
		if w == -mmCell {
			t.Fatalf("Minimax lost game %d", i)
		}
	}
}

func TestGenerateTrainingData(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(100, 1)
	if len(samples) == 0 {
		t.Fatal("expected training samples")
	}
	// The empty board with X to move should always be present.
	found := false
	for _, s := range samples {
		if s.Board == (Board{}) && s.Player == cellX {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected empty-board sample")
	}
}

func TestTrainAndEvaluate(t *testing.T) {
	t.Parallel()
	samples := GenerateTrainingData(500, 7)
	network, err := TrainNetwork(samples, 18, 20, 0.1, 7)
	if err != nil {
		t.Fatalf("TrainNetwork error: %v", err)
	}

	rp := NewRandomPlayer(8)
	result := Evaluate(network, rp, 100, 9)
	total := result.Wins + result.Draws + result.Losses
	if total != 100 {
		t.Errorf("expected 100 games, got %d", total)
	}
	// Even with few epochs, neural should win more than it loses against random.
	if result.Wins < result.Losses {
		t.Errorf("neural should outperform random: W=%d D=%d L=%d",
			result.Wins, result.Draws, result.Losses)
	}
}

func TestPlayGameTerminates(t *testing.T) {
	t.Parallel()
	r1 := NewRandomPlayer(10)
	r2 := NewRandomPlayer(20)
	for i := 0; i < 100; i++ {
		w := PlayGame(r1, r2)
		if w != cellX && w != cellO && w != cellEmpty {
			t.Fatalf("invalid winner: %d", w)
		}
	}
}
