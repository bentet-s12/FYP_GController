import os
import json
from Profiles import Profile

class GestureList:
    def __init__(self, filename="GestureList.json"):
        self.filename = filename
        self._gestures = []

        # Load gestures from file if exists, else create file
        if os.path.exists(self.filename):
            self._load()
        else:
            self._save()

    # ------------------- Internal Helpers -------------------
    def _load(self):
        """Load gesture list from file"""
        try:
            with open(self.filename, "r") as f:
                self._gestures = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load gestures: {e}")
            self._gestures = []

    def _save(self):
        """Save gesture list to file"""
        try:
            with open(self.filename, "w") as f:
                json.dump(self._gestures, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save gestures: {e}")

    # ------------------- Public Methods -------------------
    def getList(self):
        """Return full gesture list"""
        return self._gestures.copy()

    def createGesture(self, gesture_name):
        """Add a new gesture if not exists"""
        if gesture_name in self._gestures:
            print(f"[Warning] Gesture '{gesture_name}' already exists.")
            return False
        self._gestures.append(gesture_name)
        self._save()
        return True

    def deleteGesture(self, gesture_name, profiles_folder="."):
        """
        Delete a gesture from GestureList and remove it from all profile files
        profiles_folder: folder where profile JSONs are stored
        """
        # 1. Delete gesture from list
        if gesture_name not in self._gestures:
            print(f"[Warning] Gesture '{gesture_name}' does not exist in GestureList.")
            return False

        self._gestures.remove(gesture_name)
        self._save()
        print(f"[OK] Gesture '{gesture_name}' deleted from GestureList.")

        # 2. Remove from all profiles
        for filename in os.listdir(profiles_folder):
            if filename.startswith("profile_") and filename.endswith(".json"):
                profile_path = os.path.join(profiles_folder, filename)
                try:
                    profile = Profile.readFile(profile_path)
                    removed = False
                    for action in profile.getActionList()[:]:  # copy to safely remove
                        if action.getGName() == gesture_name:
                            profile.getActionList().remove(action)
                            removed = True
                            print(f"[Profile {profile.getProfileID()}] Removed action '{action.getName()}' mapped to deleted gesture.")

                    if removed:
                        profile.writeFile(profile_path)

                except Exception as e:
                    print(f"[Error] Could not update profile {filename}: {e}")

        return True

    def getGesture(self, gesture_name):
        """Return gesture if exists, else None"""
        return gesture_name if gesture_name in self._gestures else None

    def equals(self, other_list):
        """Check if this gesture list equals another list"""
        if not isinstance(other_list, list):
            return False
        return sorted(self._gestures) == sorted(other_list)

    def setGestures(self, new_list):
        """Replace current gesture list with a new one"""
        if not isinstance(new_list, list):
            raise ValueError("Input must be a list")
        self._gestures = list(set(new_list))  # remove duplicates
        self._save()

    # Optional utility
    def exists(self, gesture_name):
        """Check if gesture exists"""
        return gesture_name in self._gestures

if __name__ == "__main__":
    Glist = GestureList()

    # Create new gestures
    Glist.createGesture("fist")
    Glist.createGesture("palm")
    
    # Get full list
    print("All gestures:", Glist.getList())

    # Delete a gesture
    Glist.deleteGesture("fist")
    print("After deletion:", Glist.getList())

    # Check equality
    print("Equals ['palm']?", Glist.equals(["palm"]))

    # Replace gestures
    Glist.setGestures(["wave", "thumbs_up"])
    print("After set:", Glist.getList())
