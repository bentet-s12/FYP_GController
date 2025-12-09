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
        if key in ("left", "right", "middle"):# mouse hold
            while self.holdvalue:
                pydirectinput.mouseDown(button=self._key_pressed)
                time.sleep(0.05)
            pydirectinput.mouseUp(button=key)
        else:
            while self.holdvalue:
                pydirectinput.keyDown(key)
                time.sleep(0.5)

            pydirectinput.keyUp(key)

        print(f"[Hold Thread] Stopped {key}")
        self._is_holding = False
        
    def stopHold(self):
        """Force stop any ongoing hold"""
        self.holdvalue = False
        if self._is_holding:
            self._is_holding = False
        # keyUp to guarantee release
        if self._key_pressed not in ("left", "right", "middle"):
            pydirectinput.keyUp(self._key_pressed)
        else:
            pydirectinput.mouseUp(button=self._key_pressed)
    # ----- MAIN ACTION -----
    def useAction(self, ActionName):
        if ActionName != self._name:
            # If this is a different action, stop hold if needed
            if self._is_holding:
                self.stopHold()
            return

        # --- CLICK ---
        if self._input_type == "Click":
            if self._key_pressed in ("left", "right", "middle"):
                pydirectinput.click(button=self._key_pressed)
            else:
                pydirectinput.press(self._key_pressed)
            return

        # --- HOLD ---
        if self._input_type == "Hold":
            # Start hold only if not already holding
            if not self._is_holding:
                self.holdvalue = True  # internal only
                self._is_holding = True
                self._thread = threading.Thread(
                    target=self._hold_loop,
                    daemon=True
                )
                self._thread.start()


if __name__ == "__main__":
    # --- Create some actions ---
    action_hold = Actions("MoveForward", "w", "Hold")
    action_click = Actions("Jump", "space", "Click")
    action_mouse = Actions("Shoot", "left", "Click")

    all_actions = [action_hold, action_click, action_mouse]

    # --- Simulate triggering actions ---
    print("Starting test. Press Ctrl+C to stop.")

    try:
        while True:
            cmd = input("Type action to trigger (MoveForward / Jump / Shoot / None): ").strip()

            for action in all_actions:
                action.useAction(cmd)  # automatically handles hold and stop if gesture changes

            # Show internal state for debugging
            for action in all_actions:
                print(f"{action.getName()} | is_holding: {action._is_holding}")
            print("---")

    except KeyboardInterrupt:
        print("Exiting test, stopping all holds...")
        for action in all_actions:
            action.stopHold()
