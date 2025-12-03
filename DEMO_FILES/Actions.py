from pyKey import pressKey, releaseKey, press
import time

class Actions:
    def __init__(self, name, key_pressed=None, input_type=None):
        self._name = name
        self._key_pressed = key_pressed
        self._input_type = input_type
        self.holdvalue = False
        self._is_holding = False

#getters/setters
        
    def getName(self):
        return self._name

    def getKeyPressed(self):
        return self._key_pressed

    def getInputType(self):
        return self._input_type

    def setName(self, new_name):
        self._name = new_name

    def setKeyPressed(self, new_key):
        self._key_pressed = new_key

    def setInputType(self, new_input_type):
        self._input_type = new_input_type

    def toDict(self):
        return {
            "name": self._name,
            "key_pressed": self._key_pressed,
            "input_type": self._input_type
        }

    @staticmethod
    def fromDict(d):
        return Actions(
            name=d["name"],
            key_pressed=d["key_pressed"],
            input_type=d["input_type"]
        )
#functions

    def compareName(self, name_check) -> bool:
        return self._name == name_check

    def setholdvalue(self, newholdvalue):
        self.holdvalue = newholdvalue

    def useAction(self, ActionName):
        if not self.compareName(ActionName):
            return

        # CLICK
        if self._input_type == "Click":
            press(key=self._key_pressed, sec=0.1)
            return

        # HOLD
        if self._input_type == "Hold":
            while self.holdvalue:
                pressKey(self._key_pressed)
                self._is_holding = True
                print(f"Holding {self._key_pressed}")
                if not self.holdvalue and self._is_holding:
                    releaseKey(self._key_pressed)
                    self._is_holding = False
                    print(f"Released {self._key_pressed}")


#for testing purposes
if __name__ == "__main__":
    action1 = Actions("tab", "w", "Hold")

    while True:
        cmd = input("Type 'hold' to hold key, 'release' to release: ").lower()
        if cmd == "hold":
           action1.setholdvalue(True)
        elif cmd == "release":
           action1.setholdvalue(False)

        action1.useAction("tab")
    
