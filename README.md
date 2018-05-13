# Machine Learning project 2017-2018: Dots and boxes

This is the implementation of a dots and boxes agent by Timothy Werquin and Florian Van Heghe for the Machine Learning: Project
 course of 2017-2018 at KU Leuven.
 
 ## Requirements
 
This project has been written in Python 3.6 and has the following dependencies:
  * Numpy
  * Matplotlib
  * Tensorflow

These can be installed using pip:

```
 $ pip install -r requirements.txt
```

## Overview

The project contains a tabular Q-player that can only play on a fixed size board. It is used to benchmark how well our NN approach learns Q-values and as an opponent to train against. The Recurrent Neural Network based Q-player is the player that has been implemented, it can train and play locally or the agent can be used with websockets to play the provided [Dots and Boxes game ](https://github.com/wannesm/dotsandboxes). While training the trained model will be written periodically in <code>/model-rnn</code>. When using the agent, playing or resuming training with the regular player the latest model from <code>/model-rnn</code> will be loaded. Different options for the player can be found at the top of <code>dotsandboxesplayer.py</code>, these include whether to train or play, board size, amount of games, opponent and other parameters.

 ## Executing

 * Tabular Q-player: ```$ ./dotsandboxesgame.py```
 * NN-based Q-player: ```$ ./dotsandboxesplayer.py```
 * Agent using web-sockets: ```$ ./dotsandboxesagent.py <port>```

