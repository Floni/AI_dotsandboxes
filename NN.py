"""
Helpers for createing NN's
"""
import tensorflow as tf

class Dense:
    """
        Dense layer with activation function (or None for linear)
    """
    def __init__(self, name, input_size, output_size, activation=None):
        self.activation = activation
        self.name = name

        self.W = tf.get_variable(name + "-W", [input_size, output_size])
        self.b = tf.get_variable(name + "-b", [1, output_size],
                                 initializer=tf.zeros_initializer)

    def __call__(self, input):
        output = tf.matmul(input, self.W) + self.b

        if self.activation is not None:
            output = self.activation(output)

        return output

def create_fully_connected(name, input_size, layers):
    """
        Creates a fully connected neural network
         that takes an input of input_size
         and applies each layer to it.
        A layer is a tuple (activation function, output_size)
    """
    layer_objs = []
    last_size = input_size
    for idx, (act, size) in enumerate(layers):
        layer_objs.append(Dense(name + "-FC-" + str(idx), last_size, size, act))
        last_size = size

    def apply(input):
        last_l = input
        for obj in layer_objs:
            last_l = obj(last_l)
        return last_l
    return apply