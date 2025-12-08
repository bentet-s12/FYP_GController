class ActionManager:
    def __init__(self):
        self.actions = {}

    def add_action(self, gesture_label, action_obj):
        self.actions[gesture_label] = action_obj

    def trigger(self, gesture_label):
        if gesture_label not in self.actions:
            return  # no mapped action
        action = self.actions[gesture_label]
        action.useAction(action.getName())
