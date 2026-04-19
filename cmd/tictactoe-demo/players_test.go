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

	"github.com/wlbr/neuro/neural"
)

func TestRandomPlayerName(t *testing.T) {
	t.Parallel()
	p := NewRandomPlayer(1)
	if got := p.Name(); got != "Random" {
		t.Errorf("Name() = %q, want \"Random\"", got)
	}
}

func TestMinimaxPlayerName(t *testing.T) {
	t.Parallel()
	p := MinimaxPlayer{}
	if got := p.Name(); got != "Minimax" {
		t.Errorf("Name() = %q, want \"Minimax\"", got)
	}
}

func TestNeuralPlayerName(t *testing.T) {
	t.Parallel()
	net, _ := neural.NewNetworkWithSeed(boardSize, 4, boardSize, 0.1, 1)
	p := &NeuralPlayer{network: net}
	if got := p.Name(); got != "Neural" {
		t.Errorf("Name() = %q, want \"Neural\"", got)
	}
}

func TestRandomPlayerChoosesEmptyCell(t *testing.T) {
	t.Parallel()
	p := NewRandomPlayer(42)
	b := Board{cellX, cellO, cellX, cellO, cellEmpty, cellEmpty, cellX, cellO, cellX}
	for i := 0; i < 50; i++ {
		move := p.ChooseMove(b, cellX)
		if b[move] != cellEmpty {
			t.Fatalf("RandomPlayer chose occupied cell %d", move)
		}
	}
}

func TestMinimaxChoosesWinningMove(t *testing.T) {
	t.Parallel()
	mm := MinimaxPlayer{}
	// X can win immediately by playing position 2.
	b := Board{cellX, cellX, cellEmpty, cellO, cellO, cellEmpty, cellEmpty, cellEmpty, cellEmpty}
	move := mm.ChooseMove(b, cellX)
	if move != 2 {
		t.Errorf("expected winning move 2, got %d", move)
	}
}

func TestMinimaxBlocksOpponent(t *testing.T) {
	t.Parallel()
	mm := MinimaxPlayer{}
	// O must block X from winning at position 2.
	b := Board{cellX, cellX, cellEmpty, cellO, cellEmpty, cellEmpty, cellEmpty, cellEmpty, cellEmpty}
	move := mm.ChooseMove(b, cellO)
	if move != 2 {
		t.Errorf("expected blocking move 2, got %d", move)
	}
}

func TestMinimaxDrawsAgainstItself(t *testing.T) {
	t.Parallel()
	mm1 := MinimaxPlayer{}
	mm2 := MinimaxPlayer{}
	w := PlayGame(mm1, mm2)
	if w != cellEmpty {
		t.Errorf("two minimax players should draw, got winner %d", w)
	}
}

func TestNeuralPlayerChoosesValidMove(t *testing.T) {
	t.Parallel()
	net, _ := neural.NewNetworkWithSeed(boardSize, 18, boardSize, 0.1, 1)
	np := &NeuralPlayer{network: net}

	b := Board{cellX, cellO, cellEmpty, cellEmpty, cellX, cellEmpty, cellO, cellEmpty, cellEmpty}
	move := np.ChooseMove(b, cellX)
	if b[move] != cellEmpty {
		t.Fatalf("NeuralPlayer chose occupied cell %d", move)
	}
	// Move must be one of the empty cells.
	empty := b.emptyCells()
	found := false
	for _, e := range empty {
		if e == move {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("NeuralPlayer returned move %d which is not in emptyCells %v", move, empty)
	}
}

func TestNeuralPlayerCompletesGame(t *testing.T) {
	t.Parallel()
	net, _ := neural.NewNetworkWithSeed(boardSize, 18, boardSize, 0.1, 7)
	np := &NeuralPlayer{network: net}
	rp := NewRandomPlayer(99)
	for i := 0; i < 50; i++ {
		w := PlayGame(np, rp)
		if w != cellX && w != cellO && w != cellEmpty {
			t.Fatalf("invalid winner: %d", w)
		}
	}
}
