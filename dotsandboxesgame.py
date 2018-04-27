#!/usr/bin/env python3
"""
Implements the game for training and evaluation
"""

import random
import pickle

import time
import os.path

# some constants for board representation
VERT = 0
HORZ = 1
ACTIONS = ["v", "h"]
ACTION_N = {"v": 0, "h": 1}

# rewwards:
REWARD_CHEAT = -500

REWARD_WIN = 100
REWARD_LOSE = -100

REWARD_TIE = 0
REWARD_PLAY = 0
REWARD_BOX = 0

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

class Board:
    """Board representation used in game

    store the board as a [nb_rows+1][nb_cols+1][3] array.
    where the inner array is [VERT, HORZ, CAPTURED_PLAYER]
    """
    def __init__(self, nb_rows, nb_cols):
        """Create a new empty board"""
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        # zero for every dot
        # [vertical, horizontal, captured player]
        self.state = [[[0, 0, 0] for _ in range(nb_cols+1)] for _ in range(nb_rows+1)]

        self.scores = [0, 0]

    def copy(self):
        """Creates a new board containing the same data as the current board"""
        ret = Board(self.nb_rows, self.nb_cols)
        for row in range(self.nb_rows+1):
            for col in range(self.nb_cols+1):
                ret.state[row][col][0] = self.state[row][col][0]
                ret.state[row][col][1] = self.state[row][col][1]
        ret.scores = self.scores.copy()
        return ret

    def capture(self, row, col, player=None):
        """Let player capture the given cell"""
        if player is None:
            col = row[1]
            row = row[0]
        self.state[row][col][2] = player
        self.scores[player-1] += 1

    def get(self, row, col, dir=None):
        """Returns the value at the given edge position"""
        #if dir is None:
        #    dir = col
        #    col = row[1]
        #    row = row[0]
        return self.state[row][col][dir]

    def get_owner(self, row, col=None):
        """Returns the owner of the given cell position"""
        if col is None:
            col = row[1]
            row = row[0]
        return self.state[row][col][2]

    def set(self, row, col_dir, dir_val, value=None):
        """Sets the given edge to the given value"""
        if value is None:
            value = dir_val
            col_dir = row[1]
            row = row[0]

        assert 0 <= row <= self.nb_rows
        assert 0 <= col_dir <= self.nb_cols
        assert row != self.nb_rows or dir_val == HORZ
        assert col_dir != self.nb_cols or dir_val == VERT
        assert 0 <= value < 3

        self.state[row][col_dir][dir_val] = value

    def do_move(self, row, col, d, player):
        cheated = False
        captured = False

        if (row == self.nb_rows and d != HORZ) or \
            (col == self.nb_cols and d != VERT) or \
            self.get(row, col, d) != 0:

            cheated = True
        else:
            self.set(row, col, d, player)
            # check captured box
            if d == HORZ:
                if row > 0 and \
                    self.get(row-1, col, VERT) != 0 and \
                    self.get(row-1, col+1, VERT) != 0 and \
                    self.get(row-1, col, HORZ) != 0 and \
                    self.get(row, col, HORZ) != 0:
                    captured = True
                    self.capture(row-1, col, player)
                if row < self.nb_rows and \
                    self.get(row, col, VERT) != 0 and \
                    self.get(row, col+1, VERT) != 0 and \
                    self.get(row, col, HORZ) != 0 and \
                    self.get(row+1, col, HORZ) != 0:
                    captured = True
                    self.capture(row, col, player)
            if d == VERT:
                if col > 0 and \
                    self.get(row, col-1, VERT) != 0 and \
                    self.get(row, col, VERT) != 0 and \
                    self.get(row, col-1, HORZ) != 0 and \
                    self.get(row+1, col-1, HORZ) != 0:
                    captured = True
                    self.capture(row, col-1, player)
                if col < self.nb_cols and \
                    self.get(row, col, VERT) != 0 and \
                    self.get(row, col+1, VERT) != 0 and \
                    self.get(row, col, HORZ) != 0 and \
                    self.get(row+1, col, HORZ) != 0:
                    captured = True
                    self.capture(row, col, player)
        return cheated, captured

    def has_ended(self):
        return self.scores[0] + self.scores[1] == self.nb_cols * self.nb_rows

    def get_winner(self):
        if self.scores[0] > self.scores[1]:
            return 1
        elif self.scores[1] > self.scores[0]:
            return 2
        return 0

    def get_score(self, player):
        return self.scores[player-1]

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret = ""
        for row in range(self.nb_rows+1):
            line1 = ""
            line2 = ""
            for col in range(self.nb_cols+1):
                line1 += "+"
                h = self.get(row, col, HORZ)
                v = self.get(row, col, VERT)
                o = self.get_owner(row, col)

                line1 += " -="[h]
                line2 += " |I"[v]
                line2 += " 12"[o]
            ret += line1 + "\n"
            ret += line2 + "\n"
        ret += "scores: " + str(self.scores) + " ended: " + str(self.has_ended())
        return ret

class Player:
    """
    default random player for dots and boxes
    """

    def __init__(self, grid, time_limit=None, player=None):
        self.player = player
        self.time_limit = time_limit
        self.board_rows = grid[0]
        self.board_cols = grid[1]

        self.all_moves = []
        for row in range(self.board_rows+1):
            for col in range(self.board_cols+1):
                for dir in [0, 1]:
                    if row < self.board_rows or col < self.board_cols:
                        self.all_moves.append((row, col, dir))

    def get_possible_moves(self, board):
        """Returns a list of posible move tuples"""
        poss = []
        for row in range(self.board_rows+1):
            for col in range(self.board_cols+1):
                if row < self.board_rows and board.get(row, col, VERT) == 0:
                    poss.append((row, col, VERT))
                if col < self.board_cols and board.get(row, col, HORZ) == 0:
                    poss.append((row, col, HORZ))
        return poss

    def play(self, board, player=None, train=False):
        """
        Returns the next action,
        board is the current state,
        player is the number of this player if not defined in constructor
        """
        return random.choice(self.get_possible_moves(board))


    def reward(self, board, action, next_board, reward, done, player=None):
        """
        give the player a reward for the specified <state, action, next_state> pair
        board: the state
        action: the action taken
        next_board: the board after the action or None if done is True
        reward: the reward given
        done: True if the game was over after the given action
        """
        pass

    def summary(self):
        """
        print summary about training
        """
        pass


    def load(self):
        """
        load the state
        """
        pass

    def save(self):
        """
        Save the model state
        """
        pass


class QPlayer(Player):
    """Simple Q learning based player using a dictonary as table"""
    def __init__(self, grid, time_limit=None, player=None):
        super().__init__(grid, time_limit, player)

        # start with 100 % exploration
        self.exploration = 0.2
        self.final_exploration = 0.02
        self.exploration_fraction = 100000

        self.expl_update = (self.exploration - self.final_exploration) / self.exploration_fraction

        self.alpha = 0.1
        self.gamma = 0.9

        self.update = True

        self.step = 0

        self.Q = {}
        self.load()


    def to_state(self, board):
        """Converts a board to an immutable state tuple for use in the Q dictionary"""
        return tuple([tuple([(x[0] != 0, x[1] != 0) for x in a]) for a in board.state])

    def getQ(self, state, action):
        """Gets the Q value from the dictionary"""
        if state in self.Q:
            if action in self.Q[state]:
                return self.Q[state][action]
        return 0

    def updateQ(self, state, action, next_state, reward, done):
        """Updates the Q value after playing a move

        Args:
            state: the state the move was made in
            action: the move that was player
            next_state: the resulting state after this action (and after the other player moved)
            reward: the reward received for playing this move
            done: boolean wheter or not the game was done (used for next Q value)
        """
        max_q_next = 0

        self.exploration = max(self.exploration - self.expl_update, self.final_exploration)

        # don't get max_q_next if done:
        if not done and next_state in self.Q:
            max_q_next = max(self.Q[next_state].values())

        if state not in self.Q:
            self.Q[state] = {}

        self.Q[state][action] = self.getQ(state, action) * (1 - self.alpha) + \
                self.alpha * (reward + self.gamma * max_q_next)

        self.step += 1


    def play(self, board, player=None, train=False):
        """Plays randomly with exploration probability during training"""
        state = self.to_state(board)

        if train and random.random() < self.exploration:
            return random.choice(self.all_moves)
        else:
            actions = self.all_moves if train else self.get_possible_moves(board)
            return max(actions, key=(lambda a: self.getQ(state, a)))

    def reward(self, board, action, next_board, reward, done, player=None):
        """Rewards the player, updates the Q values"""
        if not self.update:
            return

        assert board is not None
        state = self.to_state(board)
        next_state = None
        if next_board is not None:
            next_state = self.to_state(next_board)
        self.updateQ(state, action, next_state, reward, done)

    def summary(self):
        print("step:", self.step, "exploration:", self.exploration)

    def get_save_name(self):
        name = "q_value" + str(self.board_rows) + "x" + str(self.board_cols) + ".pickle"
        return name

    def load(self):
        name = self.get_save_name()
        if os.path.exists(name):
            with open(name, "rb") as f:
                self.Q = pickle.load(f)

    def save(self):
        name = self.get_save_name()
        with open(name, "wb") as f:
            pickle.dump(self.Q, f)

class Game:
    """
    Game for training and evaluation
    """
    def __init__(self, size, player1, player2):
        """Create a new game of the given size

        Args:
            size: a size tuple for the board
            player1: a constructor for creating the first player
            player2: a constructor for creating the second player
        """
        self.size = size
        self.players = [player1(size), player2(size)]
        self.rand_start = True
        self.start_player = 0
        self.cur_player = self.start_player

        self.play_times = [0, 0]
        self.steps = [0, 0]
        self.reset()

    def reset(self):
        """Resets the game to the initial state"""
        self.board = Board(self.size[0], self.size[1])
        self.cur_player =  1 - self.cur_player if self.rand_start else self.start_player #random.randint(0, 1)
        self.ended = False
        self.winner = None
        self.cheated = False

        self.last_boards = [None, None]
        self.last_actions = [None, None]

    def copy(self):
        """Creates a copy of this game with both players set to none"""
        ret = Game(self.size, None, None)
        # from __init__
        ret.rand_start = self.rand_start
        ret.start_player = self.start_player
        ret.play_times = [self.play_times[0], self.play_times[1]]
        ret.steps = [self.steps[0], self.steps[1]]
        # from reset
        ret.board = self.board.copy()
        ret.cur_player = self.cur_player
        ret.ended = self.ended
        ret.winner = self.winner
        ret.cheated = self.cheated
        ret.last_boards = self.last_actions.copy()
        ret.last_actions = self.last_actions.copy()

    def update_board(self, row, col, d):
        """updates the board for the given move, also updates scores"""
        # update board
        cheated, captured = self.board.do_move(row, col, d, self.cur_player+1)
        if cheated:
            self.ended = True
        self.cheated = cheated

        # check win
        if self.board.has_ended():
            # game over
            self.ended = True
            self.winner = self.board.get_winner() - 1
            if self.winner < 0:
                self.winner = None

        # next player:
        if not captured and not self.ended:
            self.cur_player = 1 - self.cur_player

        return cheated, captured

    def step(self, train=False):
        """lets one player move and updates the board, also rewards the player if train is true"""
        assert not self.ended

        cur_player = self.players[self.cur_player]
        other_player = self.players[1-self.cur_player]

        # save board for rewards
        self.last_boards[self.cur_player] = self.board.copy()

        start_time = time.time()
        row, col, d = cur_player.play(self.board, self.cur_player, train)
        elapsed = time.time() - start_time
        self.play_times[self.cur_player] += elapsed
        self.steps[self.cur_player] += 1

        # save action for rewards
        self.last_actions[self.cur_player] = (row, col, d)

        # update board
        cheated, captured = self.update_board(row, col, d)

        # give reward:
        if train:
            cp = self.cur_player
            op = 1 - cp
            if cheated:
                cur_player.reward(self.last_boards[cp], self.last_actions[cp],
                                  None, REWARD_CHEAT, True, cp)
            elif self.ended:
                if self.winner is None:
                    cur_player.reward(self.last_boards[cp], self.last_actions[cp],
                                      None, REWARD_TIE, True, cp)
                    other_player.reward(self.last_boards[op], self.last_actions[op],
                                        None, REWARD_TIE, True, op)
                else:
                    w = self.winner
                    l = 1 - self.winner
                    self.players[w].reward(self.last_boards[w], self.last_actions[w],
                                           None, REWARD_WIN, True, w)
                    self.players[l].reward(self.last_boards[l], self.last_actions[l],
                                           None, REWARD_LOSE, True, l)
            else:
                if captured:
                    cur_player.reward(self.last_boards[cp], self.last_actions[cp], self.board,
                                        REWARD_BOX, False, cp)
                elif self.last_boards[op] is not None:
                    other_player.reward(self.last_boards[op], self.last_actions[op], self.board,
                                        REWARD_PLAY, False, op)

    def play_game(self):
        """Plays one game, printing the board after each move"""
        self.reset()
        while not self.ended:
            print("player: ",  self.cur_player + 1)
            self.step()
            print(self)

    def eval(self, n_games):
        """evaluates the players by playing n_games and summerizing the result"""
        start_time = time.time()
        self.train(n_games, False)
        end_time = time.time()
        elapsed = end_time - start_time
        print("total time:", elapsed, "per game:", elapsed / n_games)

    def train(self, n_games, train=True):
        """Plays n_games tracking some statistics, if train is true also rewards the players"""
        wins = [0, 0]
        draws = 0

        scores = [0, 0]

        cheats = [0, 0]
        self.play_times = [0, 0]
        self.steps = [0, 0]

        def p(x, n = 1000):
            return round(x / n * 100, 1)

        for idx in range(n_games):
            self.reset()

            while not self.ended:
                self.step(train)

            if self.cheated:
                cheats[self.cur_player] += 1
            else:
                scores[0] += self.board.get_score(1)
                scores[1] += self.board.get_score(2)
                if self.winner is None:
                    draws += 1
                else:
                    wins[self.winner] += 1

            if train and idx % 1000 == 0:
                self.players[0].summary()
                self.players[1].summary()
                print("game: ", idx, "last 1000 games:")
                print("scores: \t", scores[0], "\t", scores[1])
                print("wins: \t\t", wins[0], " (", p(wins[0]),"%)\t", wins[1], " (", p(wins[1]), "%)")
                print("cheats: \t", cheats[0], " (", p(cheats[0]), "%)\t", cheats[1], " (", p(cheats[1]), "%)")
                print("draws: ", draws, " (", p(draws), "%)")
                print("-----------------------------------------------")
                wins = [0, 0]
                draws = 0

                scores = [0, 0]

                cheats = [0, 0]

                self.players[0].save()
                self.players[1].save()
            if not train:
                printProgressBar(idx, n_games - 1)
            else:
                printProgressBar(idx % 1000, 1000 - 1, suffix=" to 1000")
        if not train:
            print("scores: \t", scores[0], "\t", scores[1])
            print("wins:   \t", wins[0], " (", p(wins[0], n_games),"%)\t", wins[1], " (", p(wins[1], n_games), "%)")
            print("cheats: \t", cheats[0], " (", p(cheats[0], n_games), "%)\t", cheats[1], " (", p(cheats[1], n_games), "%)")
            print("draws: ", draws, " (", p(draws, n_games), "%)")
            print("avg play time:\t", self.play_times[0] / self.steps[0], "\t", self.play_times[1] / self.steps[0])
            print("-----------------------------------------------")

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret = str(self.board) + "\n"
        if self.ended and self.winner is not None:
            ret += "\nwinner: " + str(self.winner+1)
        if self.ended and self.cheated:
            ret += "\nplayer " + str(self.cur_player+1) + " cheated"
        return ret

class GreedyPlayer(Player):
    """A greedy players that plays randomly excpet when it can capture a box"""
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

SELF_PLAY = False
OLDER_PLAY = False

RAND_START = True

START_FIRST = False

if __name__ == '__main__':
    g = Game((2, 2), QPlayer, GreedyPlayer)
    g.rand_start = RAND_START
    if not START_FIRST:
        g.start_player = 1

    if SELF_PLAY:
        if OLDER_PLAY:
            g.players[1] = QPlayer((2, 2))
            #g.players[1].Q = g.players[0].Q
            g.players[1].update = False
        else:
            g.players[1] = g.players[0]

    print("training")
    try:
        g.train(100000)
    except KeyboardInterrupt:
        pass
    print("evaluating")
    g.eval(1000)
    print("saving Q values")
    g.players[0].save()
    print("done")