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
	"fmt"
	"math/rand"

	"github.com/wlbr/neuro/neural"
)

// TrainingSample pairs a board position with the minimax-optimal move.
type TrainingSample struct {
	Board    Board
	Player   int
	BestMove int
}

// boardPlayerKey is a deduplication key combining a board state with the
// current player, used to eliminate duplicate training samples.
type boardPlayerKey struct {
	board  Board
	player int
}

// GenerateTrainingData plays random games and labels each encountered
// position with the minimax-optimal move.
func GenerateTrainingData(numGames int, seed int64) []TrainingSample {
	rng := rand.New(rand.NewSource(seed))
	mm := MinimaxPlayer{}
	seen := make(map[boardPlayerKey]bool)
	var samples []TrainingSample

	for g := 0; g < numGames; g++ {
		var b Board
		player := cellX
		for !b.isOver() {
			key := boardPlayerKey{b, player}
			if !seen[key] {
				bestMove := mm.ChooseMove(b, player)
				samples = append(samples, TrainingSample{b, player, bestMove})
				seen[key] = true
			}
			// Make a random move to explore varied game states.
			cells := b.emptyCells()
			move := cells[rng.Intn(len(cells))]
			b[move] = player
			player = -player
		}
	}
	return samples
}

// TrainNetwork trains a neural network to predict optimal tic-tac-toe moves.
func TrainNetwork(samples []TrainingSample, hiddenNodes, epochs int,
	learningRate float64, seed int64) (*neural.Network, error) {

	network, err := neural.NewNetworkWithSeed(boardSize, hiddenNodes, boardSize, learningRate, seed)
	if err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(seed))
	step := epochs / 4
	if step == 0 {
		step = 1
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		rng.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
		for _, s := range samples {
			inputs := s.Board.encode(s.Player)
			targets := make([]float64, boardSize)
			for i := range targets {
				targets[i] = 0.01
			}
			targets[s.BestMove] = 0.99

			if err := network.Train(inputs, targets); err != nil {
				return nil, fmt.Errorf("epoch %d: %w", epoch, err)
			}
		}
		if epoch%step == 0 || epoch == epochs {
			fmt.Printf("  Epoch %d/%d complete\n", epoch, epochs)
		}
	}
	return network, nil
}

// PlayGame plays a single game and returns the winner (cellX, cellO, or cellEmpty for draw).
func PlayGame(p1, p2 Player) int {
	var b Board
	player := cellX
	for !b.isOver() {
		if player == cellX {
			b[p1.ChooseMove(b, player)] = player
		} else {
			b[p2.ChooseMove(b, player)] = player
		}
		player = -player
	}
	return b.winner()
}

// EvalResult holds win/draw/loss counts.
type EvalResult struct {
	Wins, Draws, Losses int
}

// Evaluate plays the neural network against an opponent, alternating sides.
func Evaluate(network *neural.Network, opponent Player, games int, seed int64) EvalResult {
	np := &NeuralPlayer{network: network}
	var r EvalResult

	for g := 0; g < games; g++ {
		neuralIsX := g%2 == 0
		var w int
		if neuralIsX {
			w = PlayGame(np, opponent)
		} else {
			w = PlayGame(opponent, np)
		}

		neuralCell := cellX
		if !neuralIsX {
			neuralCell = cellO
		}
		switch {
		case w == neuralCell:
			r.Wins++
		case w == cellEmpty:
			r.Draws++
		default:
			r.Losses++
		}
	}
	return r
}

// ShowSampleGame plays and prints a game move by move.
func ShowSampleGame(p1, p2 Player) {
	var b Board
	player := cellX
	move := 0

	for !b.isOver() {
		var pos int
		if player == cellX {
			pos = p1.ChooseMove(b, player)
		} else {
			pos = p2.ChooseMove(b, player)
		}
		move++
		b[pos] = player

		name := p1.Name()
		if player == cellO {
			name = p2.Name()
		}
		fmt.Printf("  Move %d: %s (%s) → position %d\n", move, name, cellName(player), pos)
		player = -player
	}

	fmt.Println()
	fmt.Println(b.String())
	if w := b.winner(); w == cellEmpty {
		fmt.Println("  Result: Draw")
	} else {
		name := p1.Name()
		if w == cellO {
			name = p2.Name()
		}
		fmt.Printf("  Result: %s (%s) wins\n", name, cellName(w))
	}
}
