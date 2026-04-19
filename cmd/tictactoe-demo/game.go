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

import "fmt"

const boardSize = 9

const (
	cellEmpty = 0
	cellX     = 1
	cellO     = -1
)

// Board represents a tic-tac-toe board as a flat array of 9 cells.
//
//	0 | 1 | 2
//	--+---+--
//	3 | 4 | 5
//	--+---+--
//	6 | 7 | 8
type Board [boardSize]int

var winLines = [8][3]int{
	{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // rows
	{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // columns
	{0, 4, 8}, {2, 4, 6}, // diagonals
}

func (b Board) winner() int {
	for _, line := range winLines {
		if b[line[0]] != cellEmpty &&
			b[line[0]] == b[line[1]] &&
			b[line[1]] == b[line[2]] {
			return b[line[0]]
		}
	}
	return cellEmpty
}

func (b Board) emptyCells() []int {
	cells := make([]int, 0, boardSize)
	for i, c := range b {
		if c == cellEmpty {
			cells = append(cells, i)
		}
	}
	return cells
}

func (b Board) isFull() bool {
	for _, c := range b {
		if c == cellEmpty {
			return false
		}
	}
	return true
}

func (b Board) isOver() bool {
	return b.winner() != cellEmpty || b.isFull()
}

// encode converts the board to neural network inputs from the perspective
// of the given player: own piece → 1.0, opponent → −1.0, empty → 0.0.
func (b Board) encode(player int) []float64 {
	inputs := make([]float64, boardSize)
	for i, cell := range b {
		switch cell {
		case player:
			inputs[i] = 1.0
		case -player:
			inputs[i] = -1.0
		default:
			inputs[i] = 0.0
		}
	}
	return inputs
}

func (b Board) String() string {
	sym := func(c int) rune {
		switch c {
		case cellX:
			return 'X'
		case cellO:
			return 'O'
		default:
			return ' '
		}
	}
	return fmt.Sprintf(
		" %c | %c | %c\n---+---+---\n %c | %c | %c\n---+---+---\n %c | %c | %c",
		sym(b[0]), sym(b[1]), sym(b[2]),
		sym(b[3]), sym(b[4]), sym(b[5]),
		sym(b[6]), sym(b[7]), sym(b[8]),
	)
}

func cellName(player int) string {
	if player == cellX {
		return "X"
	}
	return "O"
}
