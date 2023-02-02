"""
Data structure for storing game parameters
"""
from HumanActionGetter import *

game_params = {
    "MountainCar-v0": {
        "state_size": 2,
        "action_size": 3,
        "threshold": -100,
        "max_t": 200,
        "action_getter": MountainCarHumanActionGetter
    },
    "CartPole-v1": {
        "state_size": 4,
        "action_size": 2,
        "threshold": 475,
        "max_t": 500,
        "action_getter": CartPoleHumanActionGetter
    },
    "Acrobot-v1": {
        "state_size": 6,
        "action_size": 3,
        "threshold": -100,
        "max_t": 500,
        "action_getter": AcrobotHumanActionGetter
    },
    "CarRacing-v2": {
        "state_size": (96, 96, 3),
        "action_size": 5,
        "threshold": 1000,
        "max_t": 10000,
        "action_getter": CarRacingHumanActionGetter
    },
    "LunarLander-v2": {
        "state_size": 8,
        "action_size": 4,
        "threshold": 200,
        "max_t": 1000,
        "action_getter": LunarLanderHumanActionGetter
    },
    "ALE/Breakout-v5": {
        "state_size": 128,
        "action_size": 4,
        "threshold": 1000,
        "max_t": 10000,
        "action_getter": BreakoutHumanActionGetter
    },
    "ALE/Pong-v5": {
        "state_size": 128,
        "action_size": 6,
        "threshold": 1000,
        "max_t": 10000,
        "action_getter": PongHumanActionGetter
    }
}