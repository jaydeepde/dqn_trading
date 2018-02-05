#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Trading with Deep Reinforcement Learning
"""

import trading_launcher_1d
import sys

import warnings
warnings.filterwarnings("ignore")

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 500000
    EPOCHS = 100
    STEPS_PER_TEST = 500000
    FRAME_SKIP = 1
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = .0002
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 100000
    PHI_LENGTH = 1
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "1d_net"
    FREEZE_INTERVAL = -1
    REPLAY_START_SIZE = 100
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 48
    RESIZED_HEIGHT = 1
    DEATH_ENDS_EPISODE = 'false'
    MAX_START_NULLOPS = 0
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

if __name__ == "__main__":
    trading_launcher_1d.launch(sys.argv[1:], Defaults, __doc__)
