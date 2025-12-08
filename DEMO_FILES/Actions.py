import pydirectinput
import threading
import time

class Actions:
    def __init__(self, name, key_pressed=None, input_type=None):
        self._name = name
        self._key_pressed = key_pressed
        self._input_type = input_type
        
        self.holdvalue = False
        self._is_holding = False
        self._thread = None

    # ----- Serialization -----
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

    # ----- Getters -----
    def getName(self):
        return self._name

    def getKeyPressed(self):
        return self._key_pressed

    def getInputType(self):
        return self._input_type

    def setholdvalue(self, newholdvalue):
        self.holdvalue = newholdvalue
        
    def compareName(self, name_check):
        return self._name == name_check

    def setName(self, newName):
        self._name = newName
    # ----- HOLD LOOP -----
    def _hold_loop(self):
        key = self._key_pressed
        print(f"[Hold Thread] Started repeating {key}")
        if key in ("left", "right", "middle"):  # mouse hold
            while self.holdvalue:
                pydirectinput.mouseDown(button=key)
                time.sleep(0.05)
            pydirectinput.mouseUp(button=key)
        else:
            while self.holdvalue:
                pydirectinput.keyDown(key)
                time.sleep(0.5)

            pydirectinput.keyUp(key)

        print(f"[Hold Thread] Stopped {key}")
        self._is_holding = False

    # ----- MAIN ACTION -----
    def useAction(self, ActionName):
        if ActionName != self._name:
            return

        if self._key_pressed == "left" && self._input_type == "Click":
            pydirectinput.click(button="left")
            return
        
        # --- CLICK ---
        if self._input_type == "Click":
            pydirectinput.press(self._key_pressed)
            self.holdvalue = False
            return

        # --- HOLD ---
        if self._input_type == "Hold":
            if self.holdvalue and not self._is_holding:
                self._is_holding = True
                self._thread = threading.Thread(
                    target=self._hold_loop,
                    daemon=True
                )
                self._thread.start()

            elif not self.holdvalue and self._is_holding:
                self._is_holding = False
                # loop exits by itself

if __name__ == "__main__":
    action1 = Actions("tab", "w", "Hold")

    while True:
        cmd = input("Type 'hold' / 'release': ").lower()

        if cmd == "hold":
            action1.setholdvalue(True)
        elif cmd == "release":
            action1.setholdvalue(False)

        action1.useAction("tab")


