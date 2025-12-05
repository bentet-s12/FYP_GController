import json
from Profiles import Profile
from Actions import Actions

class ProfileManager:
    def __init__(self, profileNames=None):
        self._profileNames = profileNames if profileNames else []

    @staticmethod
    def readFile(filename):
        with open(filename, "r") as f:
            data = json.load(f)

        manager = ProfileManager(data["profileNames"])
        return manager

    def writeFile(self, filename):
        data = {
            "profileNames": self._profileNames
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def addProfile(self, profileName):
        self._profileNames.append(profileName)
        self.writeFile("profileManager.json")

    def getProfiles(self):
        return self._profileNames

    def loadProfile(self, profileName):
        filename = f"profile_{profileName}.json"
        return Profile.readFile(filename)

#testing script
if __name__ == "__main__":
    # Create Profile Manager with 3 profiles
    manager = ProfileManager(["1", "2", "3"])
    manager.writeFile("profileManager.json")

    # ----- Profile 1 (0 actions) -----
    p1 = Profile("1")
    p1.writeFile("profile_1.json")

    # ----- Profile 2 (6 actions) -----
    p2 = Profile("2")
    actions_p2 = [
        Actions("Forward", "w", "Hold"),
        Actions("Backward", "s", "Hold"),
        Actions("Left", "a", "Hold"),
        Actions("Right", "d", "Hold"),
        Actions("Jump", "space", "Click"),
        Actions("Crouch", "CTRL", "Hold")
    ]
    for a in actions_p2:
        p2.addAction(a)
    p2.writeFile("profile_2.json")

    # ----- Profile 3 (2 actions) -----
    p3 = Profile("3")
    actions_p3 = [
        Actions("Shoot", "LEFT_MOUSE", "Click"),
        Actions("Aim", "RIGHT_MOUSE", "Hold")
    ]
    for a in actions_p3:
        p3.addAction(a)
        p3.writeFile("profile_3.json")

        print("Profiles 1, 2, and 3 created successfully!")

        print("\n--- Loaded Profiles and Their Actions ---")
    for profile_name in manager.getProfiles():
        profile = manager.loadProfile(profile_name)
        print(f"\nProfile ID: {profile.getProfileID()}")

        actions = profile.getActionList()

        if not actions:
            print("  (No actions)")
            continue

        for action in actions:
            print(f"  - {action.getName()} ({action.getKeyPressed()}, {action.getInputType()})")


        # ----- CALL "Jump" FROM PROFILE 2 -----
    print("\n--- Calling 'Jump' from Profile 2 ---")
    profile2 = manager.loadProfile("2")
    profile2.callfunction("Jump")
