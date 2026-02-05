import json
import os
from Profiles import Profile
from Actions import Actions

class ProfileManager:
    def __init__(self, profileNames=None, base_dir=None):
        self._profileNames = profileNames if profileNames else []
        # Default base_dir = folder that contains ProfileManager.py (same place as prototypeV2)
        self._base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

    def _path(self, filename: str) -> str:
        # Put all manager/profile json files next to ProfileManager.py
        return os.path.join(self._base_dir, filename)

    @staticmethod
    def readFile(filename, base_dir=None):
        # 1. Resolve Path
        if not os.path.isabs(filename):
            bd = base_dir or os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(bd, filename)

        data = None
        needs_repair = False

        # 2. Attempt to Read
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        print(f"[Repair] {filename} was empty. Resetting structure...")
                        needs_repair = True
                    else:
                        data = json.loads(content)
                        # Check for required keys
                        if "Profile_ID" not in data or "Actions" not in data:
                            print(f"[Repair] {filename} missing keys. Resetting...")
                            needs_repair = True
            except (json.JSONDecodeError, Exception) as e:
                print(f"[Repair] {filename} corrupted ({e}). Resetting...")
                needs_repair = True
        else:
            needs_repair = True

        # 3. Perform Repair if necessary
        if needs_repair:
            # Determine ID from filename (e.g., "profile_1.json" -> "1" or "default.json" -> "Default")
            basename = os.path.basename(filename).replace(".json", "")
            repair_id = basename.replace("profile_", "")
            if repair_id.lower() == "default": repair_id = "Default"

            # Create blank structure
            data = {
                "Profile_ID": repair_id,
                "Actions": []
            }
            
            # Save the repaired file immediately
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"[OK] {filename} has been repaired with a valid structure.")

        # 4. Return the Profile Object
        bd = base_dir or os.path.dirname(os.path.abspath(filename))
        profile = Profile(data["Profile_ID"], base_dir=bd)

        for action_dict in data.get("Actions", []):
            profile.addAction(Actions.fromDict(action_dict), autosave=False, initialize=True)

        return profile

    def writeFile(self, filename):
        data = {"profileNames": self._profileNames}
        fullpath = self._path(filename)
        with open(fullpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def deleteProfile(self, profileName):
        if profileName in self._profileNames:
            profile = self.loadProfile(profileName)
            if profile:
                profile.deleteSelf()
            self._profileNames.remove(profileName)
            self.writeFile("profileManager.json")
            print(f"[OK] Profile '{profileName}' deleted.")
            return True
        else:
            print(f"[Warning] Profile '{profileName}' not found!")
            return False

    def addProfile(self, profileName):
        if profileName in self._profileNames:
            print(f"[Warning] Profile '{profileName}' already exists!")
            return False
        if len(self._profileNames) < 9999:
            self._profileNames.append(profileName)
            self.writeFile("profileManager.json")
            new_profile = Profile(profileName)
            new_profile.writeFile()
            print(f"[OK] Profile '{profileName}' created.")
            return True
        return False

    def renameProfile(self, oldName, newName):
        if oldName not in self._profileNames:
            print(f"[Error] Profile '{oldName}' does not exist!")
            return True
        if newName in self._profileNames:
            print(f"[Error] Profile '{newName}' already exists!")
            return False

        old_filename = f"profile_{oldName}.json"
        new_filename = f"profile_{newName}.json"

        old_path = self._path(old_filename)
        new_path = self._path(new_filename)

        if not os.path.exists(old_path):
            print(f"[Error] File '{old_path}' does not exist!")
            return

        profile = self.loadProfile(oldName)
        profile._Profile_ID = newName
        profile.writeFile(new_path)
        os.remove(old_path)

        self._profileNames.remove(oldName)
        self._profileNames.append(newName)
        self.writeFile("profileManager.json")
        print(f"[OK] Profile renamed from '{oldName}' to '{newName}'.")

    def getProfileList(self):
        return self._profileNames

    def loadProfile(self, profileName):
        # Determine the filename based on the ID
        if profileName == "Default":
            filename = "default.json" # Or "Default.json" - be consistent!
        else:
            filename = f"profile_{profileName}.json"
        
        fullpath = self._path(filename)
        
        if not os.path.exists(fullpath):
            print(f"[Error] File '{fullpath}' does not exist!")
            return None
        
        return Profile.readFile(fullpath, base_dir=self._base_dir)

    def getProfile(self, profileName):
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
