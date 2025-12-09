import os
import shutil
import cv2
import subprocess
import platform
import numpy as np
import mediapipe as mp
import json


# ===================== PATH CONFIG =====================

class PathConfig:
    """
    Centralised path configuration used by the manager and collector.
    """
    def __init__(self):
        # Folder where this script lives
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Main landmark vector dataset (used by prototype / knn):
        #   data/landmarkVectors/X.npy
        #   data/landmarkVectors/y.npy
        #   data/landmarkVectors/class_names.npy
        self.LANDMARK_DIR = os.path.join(self.BASE_DIR, "data", "landmarkVectors")

        # Custom raw vectors folder (temporary scratch when recording):
        #   data/dataCustom/<gesture_name>/*.npy
        self.CUSTOM_DIR = os.path.join(self.BASE_DIR, "data", "dataCustom")

        # Ensure dirs exist
        os.makedirs(self.LANDMARK_DIR, exist_ok=True)
        os.makedirs(self.CUSTOM_DIR, exist_ok=True)


# ===================== PROFILE JSON MANAGER =====================

class ProfileJSONManager:
    """
    Minimal JSON-based manager for profile_<id>.json and profileManager.json.
    Used here to register / update gesture mappings:

      {
        "Profile_ID": "1",
        "Actions": [
          {"name": "Forward", "key_pressed": "w", "input_type": "Hold"},
          ...
        ]
      }
    """

    def __init__(self, base_dir: str, profile_id: str):
        self.base_dir = base_dir
        self.profile_id = str(profile_id)
        self.pm_path = os.path.join(base_dir, "profileManager.json")
        self.profile_path = os.path.join(base_dir, f"profile_{self.profile_id}.json")
        self._ensure_files_exist()

    # ----- helpers -----

    def _ensure_files_exist(self):
        # Ensure profileManager.json exists and includes this profile ID
        if os.path.exists(self.pm_path):
            with open(self.pm_path, "r") as f:
                data = json.load(f)
            names = data.get("profileNames", [])
            if self.profile_id not in names:
                names.append(self.profile_id)
                data["profileNames"] = names
                with open(self.pm_path, "w") as f:
                    json.dump(data, f, indent=4)
        else:
            data = {"profileNames": [self.profile_id]}
            with open(self.pm_path, "w") as f:
                json.dump(data, f, indent=4)

        # Ensure profile_<id>.json exists
        if os.path.exists(self.profile_path):
            return
        profile_data = {
            "Profile_ID": self.profile_id,
            "Actions": []
        }
        with open(self.profile_path, "w") as f:
            json.dump(profile_data, f, indent=4)

    def _load_profile(self):
        with open(self.profile_path, "r") as f:
            return json.load(f)

    def _save_profile(self, data):
        with open(self.profile_path, "w") as f:
            json.dump(data, f, indent=4)

    # ----- public API -----

    def upsert_action(self, gesture_name: str, key_pressed: str, input_type: str):
        """
        Add or update an action by gesture name.
        """
        data = self._load_profile()
        actions = data.get("Actions", [])

        for act in actions:
            if act.get("name") == gesture_name:
                act["key_pressed"] = key_pressed
                act["input_type"] = input_type
                break
        else:
            actions.append({
                "name": gesture_name,
                "key_pressed": key_pressed,
                "input_type": input_type
            })

        data["Actions"] = actions
        self._save_profile(data)
        print(f"[PROFILE] Saved/updated action '{gesture_name}' in {os.path.basename(self.profile_path)}")

    def delete_action(self, gesture_name: str):
        """
        Remove an action entry by gesture name (if present).
        """
        data = self._load_profile()
        actions = data.get("Actions", [])
        new_actions = [a for a in actions if a.get("name") != gesture_name]

        if len(new_actions) == len(actions):
            print(f"[PROFILE] No action named '{gesture_name}' found in {os.path.basename(self.profile_path)}")
            return

        data["Actions"] = new_actions
        self._save_profile(data)
        print(f"[PROFILE] Action '{gesture_name}' removed from {os.path.basename(self.profile_path)}")


# ===================== DATASET MANAGER =====================

class CombinedDatasetManager:
    """
    Manages:
      - Loading and saving landmark dataset (X.npy, y.npy, class_names.npy)
      - Appending a new gesture from dataCustom/<gesture_name> into landmarkVectors
      - Replacing an existing gesture with new vectors
      - Deleting a gesture class (with confirmation)
    """

    def __init__(self, paths: PathConfig):
        self.paths = paths

    # -------- LOAD / SAVE LANDMARK DATASET --------

    def load_landmark_dataset(self):
        """Load X, y, class_names from data/landmarkVectors."""
        X_path = os.path.join(self.paths.LANDMARK_DIR, "X.npy")
        y_path = os.path.join(self.paths.LANDMARK_DIR, "y.npy")
        classes_path = os.path.join(self.paths.LANDMARK_DIR, "class_names.npy")

        if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(classes_path)):
            print("[ERROR] Landmark dataset not found in:", self.paths.LANDMARK_DIR)
            print("        Expected X.npy, y.npy, class_names.npy")
            return None, None, None

        X = np.load(X_path)
        y = np.load(y_path)
        class_names = np.load(classes_path, allow_pickle=True)

        return X, y, class_names

    def save_landmark_dataset(self, X, y, class_names):
        """Save updated X, y, class_names back to data/landmarkVectors."""
        os.makedirs(self.paths.LANDMARK_DIR, exist_ok=True)

        X_path = os.path.join(self.paths.LANDMARK_DIR, "X.npy")
        y_path = os.path.join(self.paths.LANDMARK_DIR, "y.npy")
        classes_path = os.path.join(self.paths.LANDMARK_DIR, "class_names.npy")

        np.save(X_path, X)
        np.save(y_path, y)
        np.save(classes_path, class_names)

        print("\n[UPDATED] Saved landmark dataset to:")
        print(" ", X_path)
        print(" ", y_path)
        print(" ", classes_path)

    # -------- HELPER: LOAD VECTORS FROM dataCustom/<gesture_name> --------

    def _load_custom_vectors(self, gesture_name: str):
        """
        Load all .npy vectors from data/dataCustom/<gesture_name>.
        Returns array of shape (N, D) or None if not found.
        """
        gdir = os.path.join(self.paths.CUSTOM_DIR, gesture_name)
        if not os.path.isdir(gdir):
            print(f"[ERROR] No custom folder found for '{gesture_name}' at {gdir}")
            return None

        npy_files = sorted(f for f in os.listdir(gdir) if f.lower().endswith(".npy"))
        if not npy_files:
            print(f"[ERROR] No .npy files found for '{gesture_name}' in {gdir}")
            return None

        vecs = []
        for f in npy_files:
            vecs.append(np.load(os.path.join(gdir, f)))

        X_new = np.stack(vecs, axis=0)
        print(f"[INFO] Loaded {X_new.shape[0]} vectors for gesture '{gesture_name}' from {gdir}")
        return X_new

    def _delete_custom_folder(self, gesture_name: str):
        """
        Remove data/dataCustom/<gesture_name> after we are done with it
        (so it doesn't get reused accidentally).
        """
        gdir = os.path.join(self.paths.CUSTOM_DIR, gesture_name)
        if os.path.isdir(gdir):
            print(f"[INFO] Removing scratch custom folder: {gdir}")
            shutil.rmtree(gdir)

    # -------- CREATE: ADD NEW GESTURE --------

    def add_new_gesture(self, gesture_name: str):
        """
        Append a NEW gesture class using vectors from dataCustom/<gesture_name>.
        """
        X_old, y_old, class_names = self.load_landmark_dataset()
        if X_old is None:
            return False

        if gesture_name in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' already exists. Use EDIT instead.")
            return False

        X_new = self._load_custom_vectors(gesture_name)
        if X_new is None:
            return False

        # new label is next index
        new_label = len(class_names)
        y_new = np.full(X_new.shape[0], new_label, dtype=np.int64)

        X_comb = np.concatenate([X_old, X_new], axis=0)
        y_comb = np.concatenate([y_old, y_new], axis=0)
        class_names_new = np.append(class_names, gesture_name)

        self.save_landmark_dataset(X_comb, y_comb, class_names_new)
        # clean scratch
        self._delete_custom_folder(gesture_name)
        print(f"[DONE] Added new gesture '{gesture_name}' with label {new_label}.")
        return True

    # -------- DELETE GESTURE --------

    def delete_gesture_from_landmarks(self, gesture_name: str, delete_custom=True):
        """
        Remove all samples of a given gesture (class) from the landmark dataset.
        Re-index remaining class labels and save back.
        """
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False

        print("\n[INFO] Current classes:", class_names)

        if gesture_name not in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' not found in classes.")
            return False

        idx_to_remove = int(np.where(class_names == gesture_name)[0][0])
        print(f"[INFO] Deleting gesture '{gesture_name}' with class index {idx_to_remove}")

        keep_mask = (y != idx_to_remove)

        X_new = X[keep_mask]
        y_old_kept = y[keep_mask]

        print(f"[INFO] Samples before: {X.shape[0]}  | after delete: {X_new.shape[0]}")

        # Build new class_names (remove that entry)
        class_names_new = np.delete(class_names, idx_to_remove)

        # Remap y to 0..C-1
        kept_old_labels = sorted(set(y_old_kept.tolist()))
        old_to_new = {old: new for new, old in enumerate(kept_old_labels)}
        y_new = np.array([old_to_new[label] for label in y_old_kept], dtype=np.int64)

        print("[INFO] New classes:", class_names_new)
        print("[INFO] Label remap (old -> new):", old_to_new)

        self.save_landmark_dataset(X_new, y_new, class_names_new)

        if delete_custom:
            self._delete_custom_folder(gesture_name)

        print(f"\n[DONE] Gesture '{gesture_name}' removed from landmark dataset.")
        return True

    def delete_gesture_with_warning(self, gesture_name: str):
        """
        Ask the user for confirmation before deleting a gesture.
        """
        print(f"\nWARNING: You are about to permanently delete gesture '{gesture_name}'.")
        print("This will remove all its samples from the landmark dataset.")
        print("This action CANNOT be undone.")
        confirm = input("Type 'YES' to confirm deletion: ").strip()

        if confirm != "YES":
            print("Deletion cancelled.")
            return False

        return self.delete_gesture_from_landmarks(gesture_name, delete_custom=True)

    # -------- EDIT GESTURE (REPLACE) --------

    def edit_gesture(self, gesture_name: str, collector):
        """
        Replace an existing gesture with newly recorded samples:
          1) Confirm dangerous operation
          2) Record new vectors via collector -> dataCustom/<gesture_name>
          3) Delete old samples from landmark dataset (without deleting scratch)
          4) Append new vectors as a fresh class (same name)
        """
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False

        if gesture_name not in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' does not exist in the dataset.")
            return False

        print(f"\n=== EDIT GESTURE: {gesture_name} ===")
        print("You are about to REPLACE all existing samples for this gesture.")
        print("The old data will be permanently deleted and cannot be recovered.")
        confirm = input("Type 'YES' to continue: ").strip()

        if confirm != "YES":
            print("Edit cancelled.")
            return False

        # 1) Collect new samples
        print("\n[STEP 1] Collecting NEW samples for gesture:", gesture_name)
        collector.collect_gesture(gesture_name)

        # 2) Delete old samples, but KEEP scratch folder
        print("\n[STEP 2] Removing OLD gesture data from landmark dataset...")
        ok = self.delete_gesture_from_landmarks(gesture_name, delete_custom=False)
        if not ok:
            return False

        # 3) Add new gesture vectors (just created) into dataset
        print("\n[STEP 3] Adding NEW gesture samples into landmark dataset...")
        X_new = self._load_custom_vectors(gesture_name)
        if X_new is None:
            print("[ERROR] No new vectors found for gesture after edit. Aborting.")
            return False

        X_old2, y_old2, class_names2 = self.load_landmark_dataset()
        if X_old2 is None:
            print("[ERROR] Could not reload landmark dataset after deletion.")
            return False

        new_label = len(class_names2)
        y_new = np.full(X_new.shape[0], new_label, dtype=np.int64)
        X_final = np.concatenate([X_old2, X_new], axis=0)
        y_final = np.concatenate([y_old2, y_new], axis=0)
        class_names_final = np.append(class_names2, gesture_name)

        self.save_landmark_dataset(X_final, y_final, class_names_final)

        # 4) Clean scratch folder
        self._delete_custom_folder(gesture_name)

        print(f"\n[DONE] Gesture '{gesture_name}' has been REPLACED with new samples.")
        return True


# ===================== CUSTOM GESTURE COLLECTOR (WEBCAM) =====================

class CustomGestureCollector:
    """
    Uses MediaPipe to collect 42-D landmark vectors from webcam for a gesture.
    Saves them into data/dataCustom/<gesture_name>/*.npy (temporary),
    then the manager moves them into data/landmarkVectors.
    """

    def __init__(self, paths: PathConfig, dataset_size=200):
        self.paths = paths
        self.dataset_size = dataset_size

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    # -------- UTILITIES --------

    @staticmethod
    def open_folder(path):
        """Open folder in OS file explorer."""
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(f'explorer "{path}"')
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux
            subprocess.Popen(["xdg-open", path])

    @staticmethod
    def landmarks_to_feature_vector(hand_landmarks):
        """
        Convert MediaPipe hand landmarks into a normalized 42-dim vector.
        - 21 landmarks, each (x, y)
        - Wrist-centered, scale-normalized
        """
        xs = []
        ys = []
        for lm in hand_landmarks.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        # Translate so wrist (index 0) is origin
        xs -= xs[0]
        ys -= ys[0]

        # Scale normalize by max radius from wrist
        radii = np.sqrt(xs**2 + ys**2)
        max_r = np.max(radii)
        if max_r > 0:
            xs /= max_r
            ys /= max_r

        # Stack into 42-D vector [x0..x20, y0..y20]
        vec = np.concatenate([xs, ys], axis=0)
        return vec  # shape (42,)

    @staticmethod
    def get_hand_bbox(hand_landmarks, w, h, margin=20):
        """Return bounding box around hand in pixel coords."""
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]

        x_min = max(int(min(xs)) - margin, 0)
        y_min = max(int(min(ys)) - margin, 0)
        x_max = min(int(max(xs)) + margin, w - 1)
        y_max = min(int(max(ys)) + margin, h - 1)

        return x_min, y_min, x_max, y_max

    # -------- MAIN CAPTURE --------

    def collect_gesture(self, gesture_name: str):
        """
        Collect gesture samples and save 42-dim landmark vectors to:
        ./data/dataCustom/<gesture_name>/*.npy
        """
        gesture_folder = os.path.join(self.paths.CUSTOM_DIR, gesture_name)

        # Delete folder if it already exists
        if os.path.exists(gesture_folder):
            print(f"Folder '{gesture_folder}' exists. Deleting old data...")
            shutil.rmtree(gesture_folder)
        os.makedirs(gesture_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam.")
            return

        print(f"\n=== Gesture: {gesture_name} ===")
        print('Press "y" in the preview window to start recording.')
        print('Press "q" to cancel.\n')

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            # -------- WAIT PHASE (press 'y' to start) --------
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                    # Optional: draw bounding box on main view
                    x_min, y_min, x_max, y_max = self.get_hand_bbox(hand_lm, w, h)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f'Gesture: {gesture_name}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, 'Press "y" to start, "q" to cancel', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Gesture Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('y'):
                    print("Starting capture...")
                    break
                elif key == ord('q'):
                    print("Cancelled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # -------- CAPTURE PHASE --------
            counter = 0
            while counter < self.dataset_size:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if not results.multi_hand_landmarks:
                    cv2.putText(frame, "Hold gesture in the frame!", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Gesture Collector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Stopped early.")
                        break
                    continue

                hand_lm = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                # Optional: bounding box on main frame
                x_min, y_min, x_max, y_max = self.get_hand_bbox(hand_lm, w, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Convert to normalized feature vector (42-D)
                feature_vec = self.landmarks_to_feature_vector(hand_lm)

                # Save as .npy
                save_path = os.path.join(gesture_folder, f"{counter}.npy")
                np.save(save_path, feature_vec)

                cv2.putText(frame, f'Sample {counter+1}/{self.dataset_size}', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, 'Press "q" to stop early', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Gesture Collector", frame)
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    print("Stopped early.")
                    break

                print(f"Saved: {save_path}")
                counter += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nFinished collecting gesture '{gesture_name}'.")
        print(f"Saved to: {os.path.abspath(gesture_folder)}")

        # Optional: open folder
        self.open_folder(gesture_folder)


# ===================== TEXT MENU / ENTRY POINT =====================

def main():
    paths = PathConfig()

    # Ask which profile number to use
    profile_id = input("Enter profile number to use (e.g. 1): ").strip()
    if not profile_id:
        profile_id = "1"
    profile_mgr = ProfileJSONManager(paths.BASE_DIR, profile_id)

    manager = CombinedDatasetManager(paths)
    collector = CustomGestureCollector(paths, dataset_size=200)

    while True:
        print("\n===== Custom Gesture Manager =====")
        print("1) Create NEW gesture")
        print("2) EDIT existing gesture (replace samples)")
        print("3) DELETE gesture")
        print("4) Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            gesture_name = input("Enter NEW gesture name (e.g. Forward, Jump): ").strip()
            if gesture_name:
                key_pressed = input("Enter key to trigger (e.g. w, space, ctrl): ").strip()
                input_type = input("Enter input type (Click/Hold): ").strip()

                collector.collect_gesture(gesture_name)
                ok = manager.add_new_gesture(gesture_name)
                if ok:
                    profile_mgr.upsert_action(gesture_name, key_pressed, input_type)
            else:
                print("No gesture name entered.")

        elif choice == "2":
            gesture_name = input("Enter gesture name to EDIT (existing, e.g. Forward): ").strip()
            if gesture_name:
                key_pressed = input("Enter NEW key to trigger (e.g. w, space, ctrl): ").strip()
                input_type = input("Enter NEW input type (Click/Hold): ").strip()

                ok = manager.edit_gesture(gesture_name, collector)
                if ok:
                    profile_mgr.upsert_action(gesture_name, key_pressed, input_type)
            else:
                print("No gesture name entered.")

        elif choice == "3":
            gesture_name = input("Enter gesture name to DELETE: ").strip()
            if gesture_name:
                ok = manager.delete_gesture_with_warning(gesture_name)
                if ok:
                    profile_mgr.delete_action(gesture_name)
            else:
                print("No gesture name entered.")

        elif choice == "4":
            print("Exiting Custom Gesture Manager.")
            break
        else:
            print("Invalid option. Please choose 1–4.")


if __name__ == "__main__":
    main()
