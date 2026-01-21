import pydirectinput
import threading
import time


class Actions:
    def __init__(self, name, G_name, key_pressed=None, input_type=None, key_type=None):
        self._name = name
        self._G_name = G_name
        self._key_pressed = key_pressed
        self._input_type = input_type  # "Click" or "Hold"
        self._key_type = key_type

        self._hold_flag = False
        self._is_holding = False
        self._thread = None
        self._lock = threading.Lock()

    # ----- Serialization -----
    def toDict(self):
        return {
            "name": self._name,
            "G_name": self._G_name,
            "key_pressed": self._key_pressed,
            "input_type": self._input_type,
            "key_type": self._key_type
        }

    @staticmethod
    def fromDict(d):
        return Actions(
            name=d.get("name"),
            G_name=d.get("G_name"),  
            key_pressed=d.get("key_pressed"),
            input_type=d.get("input_type"),
            key_type=d.get("key_type")
        )
    # ----- Getters -----
    def getName(self):
        return self._name
    
    def getGName(self):
        return self._G_name

    def getKeyPressed(self):
        return self._key_pressed

    def getInputType(self):
        return self._input_type

    def getKeyType(self):
        return self._key_type

    # ----- Setters 
    def setName(self, newName):
        self._name = newName

    def setGName(self, newGName):
        self._G_name = newGName

    def SetKey(self, newKey):
        self._key_pressed = newKey

    def SetKeyType(self, newKeyType):
        self._key_type = newKeyType

    def SetDuration(self, newDuration):
        self._input_type = newDuration

    def compareName(self, Gname):
        if self._G_name == Gname:
            return self
        
    def _token_matches_this_action(self, token) -> bool:
        """
        token can be:
          - gesture name (e.g. "left_click")
          - key binding (e.g. "left", "space", "w")
          - None (meaning "execute this action")
        """
        if token is None:
            return True
        return token == self._G_name or token == self._key_pressed or token == self._name

    # ----- HOLD THREAD -----
    def _hold_loop(self):
        key = self._key_pressed
        if not key:
            with self._lock:
                self._is_holding = False
                self._hold_flag = False
            return

        # Press down ONCE
        try:
            if self._key_type == "Mouse":
                pydirectinput.mouseDown(button=key)
            else:
                pydirectinput.keyDown(key)

            # Wait until stopped
            while True:
                with self._lock:
                    if not self._hold_flag:
                        break
                time.sleep(0.01)

        finally:
            # Release ONCE
            try:
                if self._key_type == "Mouse":
                    pydirectinput.mouseUp(button=key)
                else:
                    pydirectinput.keyUp(key)
            except Exception:
                pass

            with self._lock:
                self._is_holding = False
                self._hold_flag = False

    def stopHold(self):
        """Force stop any ongoing hold + guarantee release."""
        with self._lock:
            self._hold_flag = False
            was_holding = self._is_holding

        # If holding, the thread will release in finally block.
        # But also do a "safety release" right now.
        key = self._key_pressed
        if not key:
            return

        try:
            if self._key_type == "Mouse":
                pydirectinput.mouseUp(button=key)
            else:
                pydirectinput.keyUp(key)
        except Exception:
            pass

        # Optional: wait a bit for thread to finish (non-blocking-ish)
        if was_holding and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.05)

     # ----- MAIN ACTION -----
    def useAction(self, token=None):

        if getattr(self, "_key_pressed", None) is None:
            return

        # --- bypass token check if None ---
        if token is not None:
            if token != self._G_name and token != self._key_pressed and token != self._name:
                return  # token does not match this action

        key = self._key_pressed
        if not key or not self._input_type:
            return

        # --- CLICK ---
        if self._input_type == "Click":
            try:
                if self._key_type == "Mouse":
                    pydirectinput.click(button=key)
                else:
                    pydirectinput.press(key)
            except Exception as e:
                print(f"[Actions] Click failed for {key}: {e}")
            return

        # --- DOUBLE CLICK ---
        if self._input_type == "D_Click":
            try:
                if self._key_type == "Mouse":
                    pydirectinput.click(button=key)
                    time.sleep(0.04)
                    pydirectinput.click(button=key)
                else:
                    pydirectinput.press(key)
                    time.sleep(0.04)
                    pydirectinput.press(key)
            except Exception as e:
                print(f"[Actions] D_Click failed for {key}: {e}")
            return

        # --- HOLD ---
        if self._input_type == "Hold":
            with self._lock:
                if self._is_holding:
                    return
                self._hold_flag = True
                self._is_holding = True

            self._thread = threading.Thread(target=self._hold_loop, daemon=True)
            self._thread.start()


if __name__ == "__main__":
    # Quick manual test:

    # Keyboard hold
    action_hold = Actions("hold_w", "w", "Hold", key_type="Keyboard")

    # Keyboard click
    action_click = Actions("jump", "space", "Click", key_type="Keyboard")

    # Mouse HOLD (left mouse button)
    action_mouse = Actions("shoot", "left", "Hold", key_type="Mouse")

    print("Type: hold_w / w  -> hold W")
    print("Type: shoot / left -> HOLD left mouse")
    print("Type: jump / space -> press space")
    print("Type: stop -> stop all holds")

    while True:
        cmd = input("> ").strip()

        if cmd == "stop":
            action_hold.stopHold()
            action_mouse.stopHold()
            continue

        for a in (action_hold, action_click, action_mouse):
            a.useAction(cmd)
