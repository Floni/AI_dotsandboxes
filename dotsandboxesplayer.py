#!/usr/bin/python3
"""
Implements a DQN player for dots and boxes
"""

import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

import pickle

from dotsandboxesgame import *
from NN import *
from MDRNN import *


MIN_INF = float('-inf')

# HYPER PARAMETERS:
INITIAL_EXPLORATION = 0.01
FINAL_EXPLORATION = 0.001
EXPLORATION_STEPS = 50000

DISCOUNT_GAMMA = 0.9

UPDATE_TARGET_INTERVAL = 1000

BATCH_SIZE = 32

REPLAY_BUFFER_SIZE = 10000

GRADIENT_CLIPPING_NORM = 10

LEARNING_RATE = 5e-4

DOUBLE_Q_LEARNING = True

# run parameters:
TRAIN = True

RAND_START = True
START_FIRST = False

QPLAYER = False
NN_PLAY = False
SELF_PLAY = False
GREEDY_PLAY = True

PRINT_QS = False

BOARD_SIZE = (3, 3)

TRAIN_GAMES = 100000
EVAL_GAMES = 500

# from: https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

LEFT  = 0
UPPER = 1
RIGHT = 2
LOWER = 3

LEFT_EDGE_MASK =  [1, 0, 0, 0]
UPPER_EDGE_MASK = [0, 1, 0, 0]
RIGHT_EDGE_MASK = [0, 0, 1, 0]
LOWER_EDGE_MASK = [0, 0, 0, 1]

LEFT_NEIGHBOUR_MASK  = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
UPPER_NEIGHBOUR_MASK = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
RIGHT_NEIGHBOUR_MASK = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
LOWER_NEIGHBOUR_MASK = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])

class DQNPlayer(Player):
    """
    deep Q network player for dots and boxes
    """
    def __init__(self, predictor_func, grid, time_limit=None, player=None, name=""):
        super().__init__(grid, time_limit, player)

        self.name = name

        self.predictor_func = predictor_func

            # TODO: needed?
        # the size of the state
        self.state_rows = grid[0]
        self.state_cols = grid[1]
        self.state_depth = 4

        self.exploration = INITIAL_EXPLORATION
        self.final_exploration = FINAL_EXPLORATION
        self.expl_update = (self.exploration - self.final_exploration) / EXPLORATION_STEPS

        self.gamma = DISCOUNT_GAMMA

        self.batch_size = BATCH_SIZE

        self.max_gradient = GRADIENT_CLIPPING_NORM

        self.replay_mem = deque(maxlen=REPLAY_BUFFER_SIZE)

        self.update = True

        self.null_state = self.to_state(Board(grid[0], grid[1]))

        self.last_time = time.time()
        self.reset_summary()

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.LEFT_EDGE_MASK  = tf.constant(LEFT_EDGE_MASK,  tf.float32)
            self.UPPER_EDGE_MASK = tf.constant(UPPER_EDGE_MASK, tf.float32)
            self.RIGHT_EDGE_MASK = tf.constant(RIGHT_EDGE_MASK, tf.float32)
            self.LOWER_EDGE_MASK = tf.constant(LOWER_EDGE_MASK, tf.float32)

            self.LEFT_NEIGHBOUR_MASK  = tf.constant(LEFT_NEIGHBOUR_MASK,  tf.float32)
            self.UPPER_NEIGHBOUR_MASK = tf.constant(UPPER_NEIGHBOUR_MASK, tf.float32)
            self.RIGHT_NEIGHBOUR_MASK = tf.constant(RIGHT_NEIGHBOUR_MASK, tf.float32)
            self.LOWER_NEIGHBOUR_MASK = tf.constant(LOWER_NEIGHBOUR_MASK, tf.float32)

            self.create_graph()
            self.session.run(tf.global_variables_initializer())
            self.session.run(self.target_update)
            self.load()

    def reset_summary(self):
        self.idx = 0
        self.loss_tot = 0
        self.loss_list = []

        self.reward_idx = 0
        self.avg_reward_list = []
        self.tot_reward = 0


    def average_cell(self, state):
        output = [[None] * self.state_cols for _ in range(self.state_rows)]
        inpt_array = [tf.unstack(X, self.state_cols, 1) for X in tf.unstack(state, self.state_rows, 1)]
        for row in range(self.state_rows):
            for col in range(self.state_cols):
                input = inpt_array[row][col]
                # neighbours:
                if row > 0:
                    upper = inpt_array[row-1][col]
                    upper_masked = tf.transpose(tf.matmul(self.UPPER_NEIGHBOUR_MASK, upper, transpose_b=True))
                else:
                    upper = input
                    upper_masked = self.UPPER_EDGE_MASK * upper

                if row < self.state_rows-1:
                    lower = inpt_array[row+1][col]
                    lower_masked = tf.transpose(tf.matmul(self.LOWER_NEIGHBOUR_MASK, lower, transpose_b=True))
                else:
                    lower = input
                    lower_masked = self.LOWER_EDGE_MASK * lower

                if col > 0:
                    left  = inpt_array[row][col-1]
                    left_masked = tf.transpose(tf.matmul(self.LEFT_NEIGHBOUR_MASK, left, transpose_b=True))
                else:
                    left = input
                    left_masked = self.LEFT_EDGE_MASK * left

                if col < self.state_cols-1:
                    right = inpt_array[row][col+1]
                    right_masked = tf.transpose(tf.matmul(self.RIGHT_NEIGHBOUR_MASK, right, transpose_b=True))
                else:
                    right = input
                    right_masked = self.RIGHT_EDGE_MASK * right

                output[row][col] = 0.5 * (input + left_masked + upper_masked + right_masked + lower_masked)
        return tf.transpose(output, (2, 0, 1, 3))

    def get_batch(self):
        return random.sample(self.replay_mem, k=self.batch_size)

    def add_mem(self, state, action, next_state, reward, done, next_state_valid):  #state_valid, next_state_valid):
        self.replay_mem.append((state, action, next_state, reward, done, next_state_valid))

    def create_graph(self):
        """
        Creates the computation graph of the neural network
        """

        self.global_step = tf.Variable(0, name='global_NNstep', trainable=False)

        with tf.name_scope("action_prediction"):
            self.States = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, self.state_depth], name="states_in")

            with tf.variable_scope("q_network"):
                self.q_outputs = self.predictor_func(self.States)

            # no need for valid action mask as if the chosen actions are valid non valid actions are never trained.
            #self.Valid_Action_Mask = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, 2], name="valid_in")

            self.action_scores = self.average_cell(self.q_outputs) #* self.Valid_Action_Mask

        with tf.name_scope("calc_q_vals"):
            self.Next_States = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, self.state_depth], name="next_states_in")
            self.Valid_Next_Action_Mask = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, self.state_depth], name="next_valid_in")
            #test_mask = (1 - self.Valid_Next_Action_Mask) * REWARD_CHEAT # less then min reward

            # for when in final state to mask max(Q(next, a)) (if done is true, mask == 0)
            self.Next_State_Mask = tf.placeholder(tf.float32, [None], name="next_states_mask")

            if DOUBLE_Q_LEARNING:
                with tf.variable_scope("q_network", reuse=True):
                    q_next_out = self.predictor_func(self.Next_States)
                q_next_scores = self.average_cell(q_next_out) + self.Valid_Next_Action_Mask#* self.Valid_Next_Action_Mask + test_mask

                action_size = self.state_rows*self.state_cols*self.state_depth
                action_selection = tf.argmax(tf.reshape(q_next_scores, [-1, action_size]), axis=1)
                action_selection_mask = tf.reshape(tf.one_hot(action_selection, action_size), [-1, self.state_rows, self.state_cols, self.state_depth])

                with tf.variable_scope("target_network"):
                    target_out = self.predictor_func(self.Next_States)
                action_evaluation = tf.reduce_sum(self.average_cell(target_out) * action_selection_mask, axis=(1, 2, 3))
                self.target_values = action_evaluation * self.Next_State_Mask
            else:
                # target network (see paper):
                with tf.variable_scope("target_network"):
                    self.target_outputs = self.predictor_func(self.Next_States)

                # don't update target network while learning:
                self.next_action_scores = self.average_cell(self.target_outputs) + self.Valid_Next_Action_Mask# * self.Valid_Next_Action_Mask # + test_mask

                self.target_values = tf.reduce_max(self.next_action_scores, axis=(1, 2, 3)) * self.Next_State_Mask

            self.Rewards = tf.placeholder(tf.float32, [None], name="rewards_in")

            self.future_rewards = self.Rewards + self.gamma * self.target_values

        with tf.name_scope("calc_loss"):
            # mask for chosen action
            self.Action_Mask = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, self.state_depth], name="action_mask")
            # only one action mask is '1' so sum selects this score:
            self.masked_action_scores = tf.reduce_sum(self.action_scores * self.Action_Mask, axis=(1, 2, 3))

            self.temp_diff = self.masked_action_scores - tf.stop_gradient(self.future_rewards)
            self.td_loss = tf.reduce_mean(huber_loss(self.temp_diff))
            tf.summary.scalar('td_loss', self.td_loss)

            #q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            #self.reg_loss = self.reg_param * tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in q_net_vars])
            #tf.summary.scalar('reg_loss', self.reg_loss)
            self.loss = self.td_loss #+ self.reg_loss
            #tf.summary.scalar('loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE) #tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient)

            #for i, (grad, var) in enumerate(gradients):
            #    if grad is not None:
            #        gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope("update_target"):
            self.target_update = []

            q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
            for v_q, v_target in zip(sorted(q_vars, key=lambda v: v.name), sorted(target_vars, key=lambda v: v.name)):
                update_op = v_target.assign(v_q)
                self.target_update.append(update_op)

            self.target_update = tf.group(*self.target_update)

        #with tf.name_scope("train_pickle"):
        #    self.Correct_Qs = tf.placeholder(tf.float32, [None, self.state_rows, self.state_cols, 2], name="CorrectQs")
        #    # calc loss
        #    self.Q_loss = tf.losses.mean_squared_error(self.Correct_Qs, self.action_scores)
        #    # optimize loss
        #    self.Q_train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(self.Q_loss)
        #    # update target network
        #    self.target_set = []
        #    q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
        #    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
        #    for v_q, v_target in zip(q_vars, target_vars):
        #        update_op = v_target.assign(v_q)
        #        self.target_set.append(update_op)
        #    self.Q_target_set = tf.group(*self.target_set)

        with tf.name_scope("save_model"):
            q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
            self.Saver = tf.train.Saver(q_vars + target_vars + [self.global_step], filename=self.get_save_name(), max_to_keep=3)

        with tf.name_scope("reward_summary"):
            self.avg_reward = tf.placeholder(tf.float32, name='avg_reward')
            self.avg_reward_var = tf.get_variable("avg_reward_var", shape=())
            tf.summary.scalar('Last 100 avg. reward', self.avg_reward_var)
            self.avg_reward_op = self.avg_reward_var.assign(self.avg_reward)
            self.noop = tf.no_op()

        self.sum_merged = tf.summary.merge_all()
        self.sum_writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())

    def updateQ(self, state, action, next_state, reward, done):
        #state_valid = self.get_valid_mask(state)
        next_state_valid = None
        if not done:
            next_state_valid = self.get_valid_mask(next_state)

        self.add_mem(state, action, next_state, reward, done, next_state_valid) #, state_valid, next_state_valid)
        if len(self.replay_mem) < self.batch_size:
            return

        batch = self.get_batch()

        batch_states = []
        #batch_valid_mask = []
        batch_next_states = []
        batch_next_valid_mask = []
        batch_rewards = []

        batch_action_mask = np.zeros((self.batch_size, self.state_rows, self.state_cols, self.state_depth))
        batch_next_state_mask = np.zeros(self.batch_size)

        for idx, (state, action, next_state, reward, b_done, next_state_valid) in enumerate(batch): #, state_valid, next_state_valid)
            batch_states.append(state)
            #batch_valid_mask.append(state_valid)
            batch_rewards.append(reward)

            co, oco = self.to_state_coords(*action)
            batch_action_mask[idx][co[0]][co[1]][co[2]] = 1
            if oco is not None:
                batch_action_mask[idx][oco[0]][oco[1]][oco[2]] = 1

            if not b_done:
                batch_next_state_mask[idx] = 1
                batch_next_states.append(next_state)
                batch_next_valid_mask.append(next_state_valid)
            else:
                batch_next_states.append(self.null_state)
                batch_next_valid_mask.append(self.null_state)

        do_summary = self.idx % 100 == 0
        avg_reward_cur = 0
        if do_summary:
            avg_reward_cur = self.tot_reward / self.reward_idx if self.reward_idx != 0 else 0
            self.tot_reward = 0
            self.reward_idx = 0

        _summary, _, _loss, _train_step, test1 = self.session.run([
            self.sum_merged if do_summary else self.noop,
            self.avg_reward_op if do_summary else self.noop,
            self.loss,
            self.train_op,
            self.target_values
        ], feed_dict={
            self.States: batch_states,
            #self.Valid_Action_Mask: batch_valid_mask,
            self.Next_States: batch_next_states,
            self.Valid_Next_Action_Mask: batch_next_valid_mask,
            self.Action_Mask: batch_action_mask,
            self.Rewards: batch_rewards,
            self.Next_State_Mask: batch_next_state_mask,
            self.avg_reward: avg_reward_cur
        })

        #print(test1)
        #print(test2)

        if do_summary:
            self.sum_writer.add_summary(_summary, global_step=self.global_step.eval(self.session))

        # update target:
        if self.idx % UPDATE_TARGET_INTERVAL == 0:
            self.session.run(self.target_update)

        # update exploration:
        self.exploration = max(self.final_exploration, self.exploration - self.expl_update)

        # for summary:
        self.tot_reward += reward
        self.reward_idx += 1

        self.idx += 1
        self.loss_tot += _loss
        if self.idx % 100 == 0:
            self.loss_list.append(self.loss_tot / 100)
            self.loss_tot = 0


    def to_state(self, board):
        ret = np.zeros((self.state_rows, self.state_cols, self.state_depth))
        for row in range(board.nb_rows+1):
            for col in range(board.nb_cols+1):
                h = 1 if board.get(row, col, HORZ) != 0 else 0
                v = 1 if board.get(row, col, VERT) != 0 else 0

                if row < self.state_rows and col < self.state_cols:
                    ret[row][col][UPPER] = h
                    ret[row][col][LEFT] = v

                if row > 0 and col < self.state_cols:
                    ret[row-1][col][LOWER] = h
                if col > 0 and row < self.state_rows:
                    ret[row][col-1][RIGHT] = v
        return ret

    def get_valid_mask(self, state):
        poss = np.zeros((self.state_rows, self.state_cols, self.state_depth))
        for row in range(self.state_rows):
            for col in range(self.state_cols):
                for edge in range(self.state_depth):
                    if state[row][col][edge] == 0:
                        poss[row][col][edge] = 0
                    else:
                        poss[row][col][edge] = MIN_INF
        return poss


    def getQs(self, state):
        return self.session.run(self.action_scores, feed_dict={
            self.States: state.reshape((1, self.state_rows, self.state_cols, self.state_depth)),
            #self.Valid_Action_Mask: self.get_valid_mask(state).reshape((1, self.state_rows, self.state_cols, 2))
        })[0]


    def to_state_coords(self, row, col, orient):
        if row < self.state_rows and col < self.state_cols:
            oo = 1 - orient
            other = None if row == 0 or col == 0 else (row - orient, col - oo, 2 + orient)

            return (row, col, orient), other

        if row >= self.state_rows and col < self.state_cols:
            row -= 1
        elif col >= self.state_cols and row < self.state_rows:
            col -= 1
        else:
            raise Exception("Illegal move: " + str(row) + " " + str(col))

        return (row, col, 2 + orient), None


    def play(self, board, player=None, train=False):
        state = self.to_state(board)
        Q_values = self.getQs(state)

        if train and random.random() < self.exploration:
            return random.choice(self.get_possible_moves(board))
        else:
            actions = self.get_possible_moves(board)

            def map_func(act):
                c, _ = self.to_state_coords(*act)
                return Q_values[c[0]][c[1]][c[2]]
            return max(actions, key=map_func)

    def reward(self, board, action, next_board, reward, done, player=None):
        if not self.update:
            return
        state = self.to_state(board)
        next_state = None
        if next_board is not None:
            next_state = self.to_state(next_board)
        self.updateQ(state, action, next_state, reward, done)


    # def tuple_to_state(self, tpl):
    #     ret = np.zeros((self.state_rows, self.state_cols, 2))
    #     for r in range(self.state_rows):
    #         for c in range(self.state_cols):
    #             for o in [0, 1]: # TODO: -1 for illegal positions
    #                 if tpl[r][c][o]:
    #                     ret[r][c][o] = 1.0
    #     return ret

    # def map_to_output(self, Qmap):
    #     ret = np.zeros((self.state_rows, self.state_cols, 2))
    #     for r in range(self.state_rows):
    #         for c in range(self.state_cols):
    #             for o in [0, 1]:
    #                 if (r, c, o) in Qmap:
    #                     ret[r][c][o] = Qmap[(r, c, o)]
    #     return ret

    # def learn_from_pickle(self, pickle_name, n_iters):
    #     Qs = None
    #     with open(pickle_name, "rb") as f:
    #         Qs = pickle.load(f)

    #     states = list(Qs.keys())

    #     tot_loss = 0

    #     for idx in range(n_iters):
    #         # generate batch
    #         batch = random.sample(states, self.batch_size)
    #         batch_states = [self.tuple_to_state(x) for x in batch]
    #         batch_valid_mask = [self.get_valid_mask(x) for x in batch_states]
    #         batch_labels = [self.map_to_output(Qs[x]) for x in batch]
    #         # train on batch
    #         loss, _ = self.session.run([self.Q_loss, self.Q_train_op], feed_dict={
    #             self.States: batch_states,
    #             self.Valid_Action_Mask: batch_valid_mask,
    #             self.Correct_Qs: batch_labels
    #         })
    #         tot_loss += loss

    #         if idx % 1000 == 0:
    #             print()
    #             print("step:", idx, "avg. loss:", tot_loss / 1000)
    #             tot_loss = 0
    #         printProgressBar(idx, n_iters, suffix=" to 1000")

    #     # update target network
    #     self.session.run(self.Q_target_set)

    #     print("targetQ:")
    #     print(self.map_to_output(Qs[states[0]]))

    def summary(self):
        t = time.time()
        print("Step: {}, eps: {:.3f}, avg. loss: {:.1f}, time: {:.2f}".format(self.idx, self.exploration, sum(self.loss_list[-10:]) / 10, t - self.last_time))
        self.last_time = t

    def get_save_name(self):
        return "model-" + self.name + "/weights.ckpt"

    def load(self):
        name = tf.train.latest_checkpoint("model-" + self.name + "/")
        if name is not None:
            self.Saver.restore(self.session, name)
            print("checkpoint restored")
        else:
            print("checkpoint doesn't exist")

    def save(self):
        if not self.update:
            return

        with self.graph.as_default():
            save_path = self.Saver.save(self.session, self.get_save_name(), global_step=self.global_step)
            print("saved in:", save_path)


def DQNPlayer_creator(predictor_func, name):
    def create(grid, *args, **kwargs):
        return DQNPlayer(predictor_func, grid, *args, name=name, **kwargs)
    return create


layers = [
    (tf.nn.relu, 64),
    (tf.nn.relu, 265),
    (tf.nn.relu, 32)
]

def create_network(state):
    nb_rows = state.shape[1]
    nb_cols = state.shape[2]

    input_size = nb_rows*nb_rows*4
    output_size = input_size

    input_shaped = tf.reshape(state, (-1, input_size))

    last_layer = create_fully_connected("dense-layers", input_size, layers + [(None, output_size)])

    output = last_layer(input_shaped)
    output_shaped = tf.reshape(output, (-1, nb_rows, nb_cols, 4))
    return output_shaped

def create_rnn_network(state):
    state_size = 64

    test_shape = tf.shape(state)[0]
    # TODO: make placeholder, don't require test_shape for dynamic size
    init_state = tf.zeros([test_shape, state_size])

    cells = [Dense("cell", 2 * state_size + 4, state_size, activation=tf.tanh)] * 4
    state_grid = unroll2DRNN(cells, state, init_state)

    sub_network = create_fully_connected("output", state_size, [
        (tf.nn.relu, 128),
        (tf.nn.relu, 32),
        (None, 4)
    ])

    concat_grid = sum2D(state_grid)
    output = apply2D(sub_network, concat_grid)

    return tf.transpose(output, (2, 0, 1, 3))

dqn_player = DQNPlayer_creator(create_rnn_network, "rnn")#DQNPlayer_creator(create_network, "fcc")##DQNPlayer_creator(create_network, "fcc")##create_network)



LEARN_FROM_PICKLE = False
PICKLE_NAME = "q_value2x2.pickle"

# def learn_pickle():
#     print("training from pickle")
#     p = dqn_player(BOARD_SIZE)
#     p.learn_from_pickle(PICKLE_NAME, 50000)
#     p.save()
#     print("done")

def main():
    # if LEARN_FROM_PICKLE:
    #     learn_pickle()
    #     return

    player2 = Player
    if QPLAYER:
        player2 = QPlayer
    elif SELF_PLAY:
        player2 = dqn_player
    elif NN_PLAY:
        player2 = DQNPlayer_creator(create_network, "fcc")
    elif GREEDY_PLAY:
        player2 = GreedyPlayer

    g = Game(BOARD_SIZE, dqn_player, player2)

    g.rand_start = RAND_START

    if not START_FIRST:
        g.start_player = 1

    if QPLAYER or SELF_PLAY:
        g.players[1].update = False

    if QPLAYER:
        g.players[1].exploration = 0
        g.players[1].final_exploration = 0

    if TRAIN:
        print("training")
        try:
            g.train(TRAIN_GAMES)
        except KeyboardInterrupt:
            pass

    print("evaluating")
    g.eval(EVAL_GAMES)

    if TRAIN:
        print("saving Q values")
        g.players[0].save()
        if NN_PLAY:
            g.players[1].save()

    if PRINT_QS:
        g.reset()
        g.board.set(0, 0, VERT, 1)
        g.board.set(0, 0, HORZ, 1)
        g.board.set(0, 1, VERT, 1)
        p = g.players[0]
        state = p.to_state(g.board)
        print(state)
        print("Qs:")
        print(p.getQs(state))
        print("without average:")
        print(p.session.run(p.q_outputs, feed_dict={
            p.States: state.reshape((1, p.state_rows, p.state_cols, p.state_depth)),
        })[0])

    print("done")


if __name__ == "__main__":
    main()
