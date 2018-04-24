from dotsandboxesgame import *

class GreedyPlayer(Player):
    def __init__(self, grid, time_limit = None, player=None):
        super().__init__(grid, time_limit, player)

    def play(self, board, player=None, train=False):
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                if board.get_owner(row, col) != 0:
                    continue
                open_edges = []
                if board.get(row, col, HORZ) == 0:
                    open_edges.append((row, col, HORZ))
                if board.get(row, col, VERT) == 0:
                    open_edges.append((row, col, VERT))
                if board.get(row+1, col, HORZ) == 0:
                    open_edges.append((row+1, col, HORZ))
                if board.get(row, col+1, VERT) == 0:
                    open_edges.append((row, col+1, VERT))

                if len(open_edges) == 1:
                    return open_edges[0]
        return random.choice(self.get_possible_moves(board))
