from Actions import Actions
from GestureList import GestureList
import json
import os

class Profile:
    def __init__(self, Profile_ID, ActionList=None, gesture_file="GestureList.json"):
        self._Profile_ID = Profile_ID
        self._ActionList = ActionList if ActionList is not None else []
        self.Glist = GestureList(filename=gesture_file)


    @staticmethod
    def readFile(filename):
        with open(filename, "r") as f:
            data = json.load(f)

        profile = Profile(data["Profile_ID"])

        for action_dict in data["Actions"]:
            profile.addAction(Actions.fromDict(action_dict))

        return profile

    def writeFile(self, filename):
        data = {
            "Profile_ID": self._Profile_ID,
            "Actions": [action.toDict() for action in self._ActionList]
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def deleteSelf(self):
        """Deletes this profile's JSON file."""
        filename = f"profile_{self._Profile_ID}.json"

        if os.path.exists(filename):
            os.remove(filename)
            print(f"[OK] Deleted profile data file: {filename}")
        else:
            print(f"[Warning] Profile data file does not exist: {filename}")
            
    def getProfileID(self):
        return self._Profile_ID

    def getActionList(self):
        return self._ActionList

    def addAction(self, action):
        for existing in self._ActionList:
            if existing.getGName() == action.getGName():
                print(f"[Error] Gesture is already bound to an action.")
                return False
        
        if action.getGName() in self.Glist.getList():
            self._ActionList.append(action)
            self.writeFile("profile_" + self._Profile_ID + ".json")
            return True
            
        else:
            print(f"[Error] Gesture does not exist.")
            return False
        
#tweak to use g_name instead
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
        action = getAction(actionName)
        action.SetKey(newKey)
        return True

    def editDuration(self, actionName, duration):
        action = getAction(actionName)
        action.SetDuration(duration)
        return True

if __name__ == "__main__":
    # Create a new profile
    profile = Profile("testprofile")

    # Create some actions
    action1 = Actions("Jump", "SPACEBAR", "Click")
    action2 = Actions("MoveForward", "w", "Hold")

    # Add actions to the profile
    profile.addAction(action1)
    profile.addAction(action2)

    # Print all actions
    print("Actions in profile:")
    for action in profile.getActionList():
        print(f"- {action.getName()} ({action.getKeyPressed()}, {action.getInputType()})")

    # Call an action
    print("\nCalling 'Jump' action...")
    profile.callfunction("Jump")

    # Save profile to JSON
    profile.writeFile("profile_" + profile.getProfileID() + ".json")
    print("\nProfile saved.")

    # Load the profile back from JSON
    loaded_profile = Profile.readFile("profile_" + profile.getProfileID() + ".json")
    print("\nLoaded profile actions:")
    for action in loaded_profile.getActionList():
        print(f"- {action.getName()} ({action.getKeyPressed()}, {action.getInputType()})")

