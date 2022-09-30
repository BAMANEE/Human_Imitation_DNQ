from copyreg import pickle
from pynput.keyboard import Key
import pickle

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
        self.action = 0
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            self.action = 0
        elif Key.right in keys:
            self.action = 1
        return self.action

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
        self.action = 1
        super().__init__()
    
    def get_action(self, input_state):
        keys = input_state.pressed_keys
        if Key.left in keys:
            return 0
        elif Key.right in keys:
            return 2
        else:
            return 1