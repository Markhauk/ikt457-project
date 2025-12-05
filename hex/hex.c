// Copyright (c) 2024 Ole-Christoffer Granmo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 3
#endif

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void print_board_state_from_moves(struct hex_game *hg, int moves_to_replay, int winner, int moves_back)
{
    // Temporary empty board
    int tmp_board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
    for (int i = 0; i < (BOARD_DIM+2)*(BOARD_DIM+2)*2; ++i) {
        tmp_board[i] = 0;
    }

    // Rebuild board by replaying the first moves_to_replay moves
    for (int m = 0; m < moves_to_replay; ++m) {
        int pos = hg->moves[m];
        int p   = m % 2;   // since main alternates player 0,1,0,1,...
        tmp_board[pos*2 + p] = 1;
    }

    printf("MOVES_BACK %d DATA ", moves_back);

    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++) {
            int pos = (i+1) * (BOARD_DIM + 2) + (j+1);
            int v = 0;
            if (tmp_board[pos * 2] == 1)       v = 1;  // Player 0 = X
            else if (tmp_board[pos * 2 + 1])   v = 2;  // Player 1 = O
            printf("%d ", v);
        }
    }
    printf("WINNER %d\n", winner);
    fflush(stdout);
}


void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

void hg_place_piece_based_on_tm_input(struct hex_game *hg, int player)
{
	printf("TM!\n");
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

void hg_print(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < i; j++) {
			printf(" ");
		}

		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				printf(" X");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				printf(" O");
			} else {
				printf(" ·");
			}
		}
		printf("\n");
	}
}

int main(int argc, char **argv) {
    struct hex_game hg;

    int winner = -1;

    int max_games = 10000;    // default
    if (argc > 1) {
        max_games = atoi(argv[1]);   // use the argument from Python
    }

    for (int game = 0; game < max_games; ++game) {
        hg_init(&hg);

        int player = 0;
        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);

            if (hg_winner(&hg, player, position)) {
                winner = player;
                break;
            }
            player = 1 - player;
        }

        int total_moves = BOARD_DIM*BOARD_DIM - hg.number_of_open_positions;

        // Final board (0 moves back)
        print_board_state_from_moves(&hg, total_moves, winner, 0);

        // 2 moves before the end, if possible
        if (total_moves > 2) {
            print_board_state_from_moves(&hg, total_moves - 2, winner, 2);
        }

        // 5 moves before the end, if possible
        if (total_moves > 5) {
            print_board_state_from_moves(&hg, total_moves - 5, winner, 5);
        }

        

        // If enough empty positions remain, this game was short
        int max_cells = BOARD_DIM * BOARD_DIM;
        if (hg.number_of_open_positions >= max_cells * 0.70) {
            printf("\nPlayer %d wins!\n\n", winner);
            hg_print(&hg);
        }
    }
}

