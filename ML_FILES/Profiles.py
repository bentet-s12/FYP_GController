from Actions import Actions
from GestureList import GestureList
import json
import os

class Profile:
    def __init__(self, Profile_ID, ActionList=None, gesture_file="GestureList.json", base_dir=None):
        self._Profile_ID = Profile_ID
        self._ActionList = ActionList if ActionList is not None else []

        # Base folder where profiles + GestureList.json live
        # Default: folder where this Profiles.py file is located
        self._base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

        # GestureList file should be resolved relative to base_dir
        gesture_path = self._abs(gesture_file)
        self.Glist = GestureList(filename=gesture_path)

    def _abs(self, filename: str) -> str:
        # If already absolute, keep it; else join base_dir
        return filename if os.path.isabs(filename) else os.path.join(self._base_dir, filename)

    @staticmethod
    def readFile(filename, base_dir=None):
        # Make read stable: if relative, resolve relative to Profiles.py folder (or provided base_dir)
        if not os.path.isabs(filename):
            bd = base_dir or os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(bd, filename)

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure profile created with same base_dir
        bd = base_dir or os.path.dirname(os.path.abspath(filename))
        profile = Profile(data["Profile_ID"], base_dir=bd)

        for action_dict in data.get("Actions", []):
            profile.addAction(Actions.fromDict(action_dict), autosave=False, initialize =True)

        return profile

    def writeFile(self, filename=None):
        if filename is None:
            # LOGIC: Default stays default.json, others get profile_ prefix
            if self._Profile_ID == "Default":
                filename = "Default.json"
            else:
                filename = f"profile_{self._Profile_ID}.json"
        
        filename = self._abs(filename)

        data = {
            "Profile_ID": self._Profile_ID,
            "Actions": [action.toDict() for action in self._ActionList]
            }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def deleteSelf(self):
        filename = self._abs(f"profile_{self._Profile_ID}.json")
        if os.path.exists(filename):
            os.remove(filename)
            print(f"[OK] Deleted profile data file: {filename}")
        else:
            print(f"[Warning] Profile data file does not exist: {filename}")

    def getProfileID(self):
        return self._Profile_ID

    def getActionList(self):
        return self._ActionList

    def addAction(self, action, autosave=True ,initialize = False):
        # prevent duplicate gesture bindings
        if(initialize == False):
            for existing in self._ActionList:
                if existing.getGName() == action.getGName():
                    print("[Error] Gesture is already bound to an action.")
                    return False

            # enforce gesture existence
            if action.getGName() in self.Glist.getList():
                self._ActionList.append(action)
                if autosave:
                    self.writeFile()  # uses base_dir default file
                return True
            else:
                print("[Error] Gesture does not exist.")
                return False
        else:
            self._ActionList.append(action)
            return True

    def getAction(self, actionName):
        for action in self._ActionList:
            if action.compareName(actionName):
                return action
        return None

    def callfunction(self, actionName):
        currentAction = self.getAction(actionName)
        if currentAction is not None:
            currentAction.useAction(actionName)
        else:
            print(f"Action '{actionName}' not found in profile.")
            return None

    def editAction(self, actionName, newkey):
        action = self.getAction(actionName)
        if action is None:
            return False
        action.SetKey(newkey)   # FIXED: removed newKey typo
        self.writeFile()
        return True

    def editDuration(self, actionName, duration):
        action = self.getAction(actionName)
        if action is None:
            return False
        action.SetDuration(duration)
        self.writeFile()
        return True

    def delete_action_by_name(self, action_name):
        """Delete by action.getName()"""
        for i, action in enumerate(self._ActionList):
            if action.compareName(action_name):
                del self._ActionList[i]
                self.writeFile()
                print(f"[OK] Deleted action (by name): {action_name}")
                return True
        print(f"[Error] Action '{action_name}' not found.")
        return False

    def delete_action_by_gesture(self, gesture_name):
        """Delete by action.getGName()"""
        for i, action in enumerate(self._ActionList):
            if action.getGName() == gesture_name:
                del self._ActionList[i]
                self.writeFile()
                print(f"[OK] Deleted action (by gesture): {gesture_name}")
                return True
        print(f"[Error] Gesture '{gesture_name}' not found in actions.")
        return False



if __name__ == "__main__":
    profile = Profile("testprofile")

    # --- Ensure gestures exist (CRITICAL) ---
    if "JumpGesture" not in profile.Glist.getList():
        profile.Glist._gestures.append("JumpGesture")
    if "MoveGesture" not in profile.Glist.getList():
        profile.Glist._gestures.append("MoveGesture")
    profile.Glist._save()

    # --- Create actions (CORRECT constructor usage) ---
    action1 = Actions(
        name="Jump",
        G_name="JumpGesture",
        key_pressed="space",
        input_type="Click",
        key_type="Keyboard"
    )

    action2 = Actions(
        name="MoveForward",
        G_name="MoveGesture",
        key_pressed="w",
        input_type="Hold",
        key_type="Keyboard"
    )

    # --- Add actions ---
    profile.addAction(action1)
    profile.addAction(action2)

    filename = f"profile_{profile.getProfileID()}.json"

    print("\n[Before delete]")
    for a in profile.getActionList():
        print("-", a.getName(), "| gesture:", a.getGName())

    # --- Delete ---
    print("\nDeleting JumpGesture...")
    profile.delete_action_by_gesture("JumpGesture")

    print("\n[After delete]")
    for a in profile.getActionList():
        print("-", a.getName())

    # --- Reload ---
    print("\n[Reload from disk]")
    loaded = Profile.readFile(filename)
    for a in loaded.getActionList():
        print("-", a.getName())

