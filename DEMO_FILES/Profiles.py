from Actions import Actions
import json

class Profile:
    def __init__(self, Profile_ID, ActionList=None):
        self._Profile_ID = Profile_ID
        self._ActionList = ActionList if ActionList is not None else []


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

            
    def getProfileID(self):
        return self._Profile_ID

    def getActionList(self):
        return self._ActionList

    def addAction(self, action):
        self._ActionList.append(action)
        self.writeFile("profile_" + self._Profile_ID + ".json")
        return

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
