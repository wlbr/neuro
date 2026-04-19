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
	"math/rand"

	"github.com/wlbr/neuro/neural"
)

// Player selects a move given a board state and current player identity.
type Player interface {
	ChooseMove(b Board, player int) int
	Name() string
}

// RandomPlayer picks a random empty cell.
type RandomPlayer struct {
	rng *rand.Rand
}

// NewRandomPlayer returns a RandomPlayer seeded with the given value.
func NewRandomPlayer(seed int64) *RandomPlayer {
	return &RandomPlayer{rng: rand.New(rand.NewSource(seed))}
}

// ChooseMove picks a random empty cell on the board.
func (p *RandomPlayer) ChooseMove(b Board, _ int) int {
	cells := b.emptyCells()
	return cells[p.rng.Intn(len(cells))]
}

// Name returns "Random".
func (p *RandomPlayer) Name() string { return "Random" }

// MinimaxPlayer plays optimally using the minimax algorithm.
type MinimaxPlayer struct{}

// ChooseMove returns the optimal move determined by the minimax algorithm.
func (p MinimaxPlayer) ChooseMove(b Board, player int) int {
	bestScore := -1000
	bestMove := -1
	for _, pos := range b.emptyCells() {
		next := b
		next[pos] = player
		score := minimax(next, 1, false, player)
		if score > bestScore {
			bestScore = score
			bestMove = pos
		}
	}
	return bestMove
}

// Name returns "Minimax".
func (p MinimaxPlayer) Name() string { return "Minimax" }

// minimax recursively evaluates a board position and returns a score from the
// perspective of originalPlayer. Higher scores are better for originalPlayer.
func minimax(b Board, depth int, maximizing bool, originalPlayer int) int {
	opp := -originalPlayer
	switch b.winner() {
	case originalPlayer:
		return 10 - depth
	case opp:
		return depth - 10
	}
	if b.isFull() {
		return 0
	}

	if maximizing {
		best := -1000
		for _, pos := range b.emptyCells() {
			next := b
			next[pos] = originalPlayer
			if s := minimax(next, depth+1, false, originalPlayer); s > best {
				best = s
			}
		}
		return best
	}

	best := 1000
	for _, pos := range b.emptyCells() {
		next := b
		next[pos] = opp
		if s := minimax(next, depth+1, true, originalPlayer); s < best {
			best = s
		}
	}
	return best
}

// NeuralPlayer uses a trained neural network to choose moves.
type NeuralPlayer struct {
	network *neural.Network
}

// ChooseMove queries the neural network and picks the highest-scored empty cell.
func (p *NeuralPlayer) ChooseMove(b Board, player int) int {
	inputs := b.encode(player)
	outputs, err := p.network.Query(inputs)
	if err != nil {
		panic(err)
	}

	// Pick the highest-scored empty cell.
	bestScore := -1.0
	bestMove := -1
	for _, pos := range b.emptyCells() {
		if outputs[pos] > bestScore {
			bestScore = outputs[pos]
			bestMove = pos
		}
	}
	return bestMove
}

// Name returns "Neural".
func (p *NeuralPlayer) Name() string { return "Neural" }
