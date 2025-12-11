import json
import os
from Profiles import Profile
from Actions import Actions

class ProfileManager:
    def __init__(self, profileNames=None):
        self._profileNames = profileNames if profileNames else []

    @staticmethod
    def readFile(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return ProfileManager(data["profileNames"])

    def writeFile(self, filename):
        data = {"profileNames": self._profileNames}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def deleteProfile(self, profileName):
        if profileName in self._profileNames:
            profile = self.loadProfile(profileName)
            if profile:
                profile.deleteSelf()
            self._profileNames.remove(profileName)
            self.writeFile("profileManager.json")
            print(f"[OK] Profile '{profileName}' deleted.")
        else:
            print(f"[Warning] Profile '{profileName}' not found!")

    def addProfile(self, profileName):
        if profileName in self._profileNames:
            print(f"[Warning] Profile '{profileName}' already exists!")
            return False
        if len(self._profileNames) < 4:
            self._profileNames.append(profileName)
            self.writeFile("profileManager.json")
            new_profile = Profile(profileName)
            new_profile.writeFile(f"profile_{profileName}.json")
            print(f"[OK] Profile '{profileName}' created.")
            return True
        else:
            return False
        

    def renameProfile(self, oldName, newName):
        if oldName not in self._profileNames:
            print(f"[Error] Profile '{oldName}' does not exist!")
            return
        if newName in self._profileNames:
            print(f"[Error] Profile '{newName}' already exists!")
            return

        old_filename = f"profile_{oldName}.json"
        new_filename = f"profile_{newName}.json"

        if not os.path.exists(old_filename):
            print(f"[Error] File '{old_filename}' does not exist!")
            return

        profile = self.loadProfile(oldName)
        profile._Profile_ID = newName
        profile.writeFile(new_filename)
        os.remove(old_filename)

        self._profileNames.remove(oldName)
        self._profileNames.append(newName)
        self.writeFile("profileManager.json")
        print(f"[OK] Profile renamed from '{oldName}' to '{newName}'.")

    def getProfileList(self):
        return self._profileNames

    def loadProfile(self, profileName):
        filename = f"profile_{profileName}.json"
        if not os.path.exists(filename):
            print(f"[Error] File '{filename}' does not exist!")
            return None
        return Profile.readFile(filename)

    def getProfile(self, profileName):
        """
        Returns the Profile object if it exists in manager, otherwise None.
        """
        if profileName in self._profileNames:
            return self.loadProfile(profileName)
        return None

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
    p2.writeFile("profile_2.json")  # Save once after adding all actions

    # ----- Profile 3 (2 actions) -----
    p3 = Profile("3")
    actions_p3 = [
        Actions("Shoot", "LEFT_MOUSE", "Click"),
        Actions("Aim", "RIGHT_MOUSE", "Hold")
    ]
    for a in actions_p3:
        p3.addAction(a)
    p3.writeFile("profile_3.json")  # Save once

    print("Profiles 1, 2, and 3 created successfully!\n")

    # ----- Display all profiles and actions -----
    print("--- Loaded Profiles and Their Actions ---")
    for profile_name in manager.getProfileList():
        profile = manager.getProfile(profile_name)
        if profile is None:
            continue

        print(f"\nProfile ID: {profile.getProfileID()}")
        actions = profile.getActionList()
        if not actions:
            print("  (No actions)")
            continue

        for action in actions:
            print(f"  - {action.getName()} ({action.getKeyPressed()}, {action.getInputType()})")

    # ----- CALL "Jump" FROM PROFILE 2 -----
    print("\n--- Calling 'Jump' from Profile 2 ---")
    profile2 = manager.getProfile("2")
    if profile2:
        profile2.callfunction("Jump")
    else:
        print("Profile 2 not found!")
