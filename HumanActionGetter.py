"""
Various classes for getting human actions for different games.
"""
import pickle
from pynput.keyboard import Key


class HumanActionGetter():
    def __init__(self):
        pass
    
    def get_action(self, input_state):
        pass

class ActionFromFileGetter(HumanActionGetter):
    def __init__(self, action_file):
        self.actions = pickle.load(open(action_file, "rb"))
    
    def get_action(self, input_state):
        return self.actions.popleft()

class BreakoutHumanActionGetter(HumanActionGetter):
    def __init__(self):
        self.action = 0
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            self.action = 3
        elif Key.right in keys:
            self.action = 2
        elif Key.up in keys:
            self.action = 1
        else:
            self.action = 0
        return self.action

class CartPoleHumanActionGetter(HumanActionGetter):
    def __init__(self):
        self.action = None
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            self.action = 0
        elif Key.right in keys:
            self.action = 1
        while self.action is None:
            if Key.left in keys:
                self.action = 0
            elif Key.right in keys:
                self.action = 1
        return self.action

class AcrobotHumanActionGetter(HumanActionGetter):
    def __init__(self):
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            return 0
        elif Key.right in keys:
            return 2
        return 1

class CarRacingHumanActionGetter(HumanActionGetter):
    def __init__(self):
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            return 2
        elif Key.right in keys:
            return 1
        elif Key.up in keys:
            return 3
        elif Key.down in keys:
            return 4
        return 0

class LunarLanderHumanActionGetter(HumanActionGetter):
    def __init__(self):
        self.action = 0
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.down in keys:
            self.action = 0
        elif Key.left in keys:
            self.action = 1
        elif Key.up in keys:
            self.action = 2
        elif Key.right in keys:
            self.action = 3
        return self.action

class MountainCarHumanActionGetter(HumanActionGetter):
    def __init__(self):
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            return 0
        elif Key.right in keys:
            return 2
        else:
            return 1

class PongHumanActionGetter(HumanActionGetter):
    def __init__(self):
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys and Key.up in keys:
            return 5
        elif Key.right in keys and Key.up in keys:
            return 4
        elif Key.left in keys:
            return 3
        elif Key.right in keys:
            return 2
        elif Key.up in keys:
            return 1
        else:
            return 0
