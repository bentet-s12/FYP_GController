class Actions:
    def __init__(self, name, key_pressed=None, input_type=None, pose_name=None):
        self._name = name
        self._key_pressed = key_pressed
        self._input_type = input_type
        self._pose_name = pose_name

    def compareName(self, name_check) -> bool:
        return self._name == name_check

    def getName(self):
        return self._name

    def getKeyPressed(self):
        return self._key_pressed

    def getInputType(self):
        return self._input_type

    def getPoseName(self):
        return self._pose_name

    def setName(self, new_name):
        self._name = new_name

    def setKeyPressed(self, new_key):
        self._key_pressed = new_key

    def setInputType(self, new_input_type):
        self._input_type = new_input_type

    def setPoseName(self, new_pose_name):
        self._pose_name = new_pose_name

    def useAction(self, ActionName):
        if(compareName(ActionName)):
            
