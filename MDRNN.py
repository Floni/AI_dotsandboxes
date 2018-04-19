import numpy as np
import tensorflow as tf

def unroll_direction(cell, input_grid, start_state, width, height, direction):
    state_grid = [[None] * height for _ in range(width)]
    dir_x = direction[0]
    dir_y = direction[1]
    for cur_x in range(width)[::dir_x]:
        for cur_y in range(height)[::dir_y]:
            current_input = input_grid[cur_x][cur_y]
            #current_input = tf.reshape(current_input, [-1, input_size])
            #print(current_input.shape)
            nx = cur_x - dir_x
            ny = cur_y - dir_y
            current_state_x = state_grid[cur_x-dir_x][cur_y] if 0 <= nx < width else start_state
            current_state_y = state_grid[cur_x][cur_y-dir_y] if 0 <= ny < height else start_state
            input_and_state = tf.concat([current_input, current_state_x, current_state_y], axis = 1)
            #print(input_and_state.shape)
            out_state = cell(input_and_state)

            state_grid[cur_x][cur_y] = out_state
    return state_grid

def unroll2DRNN(cells, inputs, input_state):
    width = inputs.shape[1]
    height = inputs.shape[2]
    # first split along x direction to give list [x = 0, x = 1, x = 2, ..]
    # then unstack each x along y irection to give [x = 0 [y = 0, y = 1], x = 1 [...], ...]
    input_series = [tf.unstack(X, height, 1) for X in tf.unstack(inputs, width, 1)]
    grid1 = unroll_direction(cells[0], input_series, input_state, width, height, (1, 1))
    grid2 = unroll_direction(cells[1], input_series, input_state, width, height, (-1, 1))
    grid3 = unroll_direction(cells[2], input_series, input_state, width, height, (1, -1))
    grid4 = unroll_direction(cells[3], input_series, input_state, width, height, (-1, -1))
    return [[[grid1[x][y], grid2[x][y], grid3[x][y], grid4[x][y]] for y in range(height)] for x in range(width)]

def concat2D(input_grid):
    return [[tf.concat(v, axis=1) for v in y_arr] for y_arr in input_grid]

def apply2D(cell, input_grid):
    return [[cell(v) for v in y_arr] for y_arr in input_grid]