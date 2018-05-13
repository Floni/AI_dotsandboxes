import numpy as np
import tensorflow as tf

def in_board(width, height, row, col):
    """ Checks if the given row/col are in the board matrix representation"""
    return (row + col >= height-1) and (row + col < 2*width + height) \
            and (col - row <= height) and (row - col <= height)

def unroll_direction(cell, input_grid, start_state,
                     width, height, board_rows, board_cols, direction):
    state_grid = [[None] * height for _ in range(width)]
    dir_row = direction[0]
    dir_col = direction[1]
    for row in range(height)[::dir_row]:
        for col in range(width)[::dir_col]:
            current_input = input_grid[row][col]

            nrow = row - dir_row
            ncol = col - dir_col
            current_state_row = start_state
            if in_board(board_cols, board_rows, nrow, col):
                current_state_row = state_grid[nrow][col]
            current_state_col = start_state
            if in_board(board_cols, board_rows, row, ncol):
                current_state_col = state_grid[row][ncol]

            # sum state as changing x/y shouldn't change output
            current_state = current_state_col + current_state_row

            _, out_state = cell(current_input, current_state)

            state_grid[row][col] = out_state
    return state_grid

def unroll2DRNN(cells, inputs, input_state, board_rows, board_cols):
    """ unrolls the given four cells in the four directions over the inputs"""
    height = inputs.shape[1]
    width = inputs.shape[2]

    # first split along x direction to give list [x = 0, x = 1, x = 2, ..]
    # then unstack each x along y irection to give [x = 0 [y = 0, y = 1], x = 1 [...], ...]
    input_series = [tf.unstack(X, height, 1) for X in tf.unstack(inputs, width, 1)]

    grid1 = unroll_direction(
        cells[0], input_series, input_state,
        width, height, board_rows, board_cols, (1, 1))
    grid2 = unroll_direction(
        cells[1], input_series, input_state,
        width, height, board_rows, board_cols, (-1, 1))
    grid3 = unroll_direction(
        cells[2], input_series, input_state,
        width, height, board_rows, board_cols, (1, -1))
    grid4 = unroll_direction(
        cells[3], input_series, input_state,
        width, height, board_rows, board_cols, (-1, -1))
    return [[
            [grid1[x][y], grid2[x][y], grid3[x][y], grid4[x][y]]
            for y in range(height)]
        for x in range(width)]

def concat2D(input_grid):
    return [[tf.concat(v, axis=1) for v in y_arr] for y_arr in input_grid]

def sum2D(input_grid):
    return [[tf.add_n(v) for v in y_arr] for y_arr in input_grid]

def apply2D(cell, input_grid):
    return [[cell(v) for v in y_arr] for y_arr in input_grid]