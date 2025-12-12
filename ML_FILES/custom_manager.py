import os
import shutil
import cv2
import subprocess
import platform
import numpy as np
import mediapipe as mp
import tempfile
import json
import glob


# ===================== PATH CONFIG =====================

class PathConfig:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.LANDMARK_DIR = os.path.join(self.BASE_DIR, "data", "landmarkVectors")
        os.makedirs(self.LANDMARK_DIR, exist_ok=True)

    @property
    def X_path(self):
        return os.path.join(self.LANDMARK_DIR, "X.npy")

    @property
    def y_path(self):
        return os.path.join(self.LANDMARK_DIR, "y.npy")

    @property
    def classes_path(self):
        return os.path.join(self.LANDMARK_DIR, "class_names.npy")


# ===================== PROFILE FILE OPS (YOUR SCHEMA) =====================

class ProfileFileStore:
    """
    Supports reading either:
      - profile_1.json   (preferred)
      - profile_1        (legacy)
    Always SAVES to profile_1.json (so Create will definitely update it).
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def json_path(self, profile_id: str) -> str:
        return os.path.join(self.base_dir, f"profile_{profile_id}.json")

    def legacy_path(self, profile_id: str) -> str:
        return os.path.join(self.base_dir, f"profile_{profile_id}")

    def resolve_read_path(self, profile_id: str) -> str | None:
        jp = self.json_path(profile_id)
        lp = self.legacy_path(profile_id)
        if os.path.exists(jp):
            return jp
        if os.path.exists(lp):
            return lp
        return None

    def load_profile(self, profile_id: str) -> dict | None:
        path = self.resolve_read_path(profile_id)
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # normalize schema
            if "Profile_ID" not in data:
                data["Profile_ID"] = str(profile_id)
            if "Actions" not in data or not isinstance(data["Actions"], list):
                data["Actions"] = []
            return data
        except Exception as e:
            print(f"[ERROR] Failed to read profile file for {profile_id}: {e}")
            return None

    def save_profile(self, profile_id: str, data: dict) -> None:
        # ALWAYS save into .json
        path = self.json_path(profile_id)
        data.setdefault("Profile_ID", str(profile_id))
        if "Actions" not in data or not isinstance(data["Actions"], list):
            data["Actions"] = []
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[PROFILE] Saved -> {os.path.basename(path)}")

    def ensure_profile(self, profile_id: str) -> dict:
        data = self.load_profile(profile_id)
        if data is None:
            data = {"Profile_ID": str(profile_id), "Actions": []}
            self.save_profile(profile_id, data)
        return data

    def upsert_action(self, profile_id: str, name: str, key_pressed, input_type):
        """
        Overwrite action if same 'name' exists, else append.
        """
        data = self.ensure_profile(profile_id)
        actions = data["Actions"]

        # normalize type
        it = input_type
        if isinstance(it, str):
            it = it.capitalize()
            if it not in ("Click", "Hold"):
                it = input_type

        updated = False
        for a in actions:
            if isinstance(a, dict) and a.get("name") == name:
                a["key_pressed"] = key_pressed
                a["input_type"] = it
                updated = True

        if not updated:
            actions.append({"name": name, "key_pressed": key_pressed, "input_type": it})

        data["Actions"] = actions
        self.save_profile(profile_id, data)
        print(f"[PROFILE] {'Updated' if updated else 'Added'} action '{name}' in profile_{profile_id}.json")

    def remove_action(self, profile_id: str, name: str) -> int:
        data = self.load_profile(profile_id)
        if data is None:
            return 0
        actions = data.get("Actions", [])
        if not isinstance(actions, list):
            return 0
        before = len(actions)
        actions = [a for a in actions if not (isinstance(a, dict) and a.get("name") == name)]
        removed = before - len(actions)
        if removed > 0:
            data["Actions"] = actions
            self.save_profile(profile_id, data)
        return removed

    def rename_action(self, profile_id: str, old_name: str, new_name: str) -> bool:
        data = self.load_profile(profile_id)
        if data is None:
            return False
        actions = data.get("Actions", [])
        if not isinstance(actions, list):
            return False

        old_action = None
        for a in actions:
            if isinstance(a, dict) and a.get("name") == old_name:
                old_action = a
                break
        if old_action is None:
            return False

        # If new exists, overwrite it with old fields, then remove old
        found_new = False
        for a in actions:
            if isinstance(a, dict) and a.get("name") == new_name:
                a["key_pressed"] = old_action.get("key_pressed")
                a["input_type"] = old_action.get("input_type")
                found_new = True

        actions = [a for a in actions if not (isinstance(a, dict) and a.get("name") == old_name)]
        if not found_new:
            old_action["name"] = new_name
            actions.append(old_action)

        data["Actions"] = actions
        self.save_profile(profile_id, data)
        return True

    def list_profile_ids(self) -> list[str]:
        ids = set()

        for p in glob.glob(os.path.join(self.base_dir, "profile_*.json")):
            base = os.path.basename(p)
            if base.startswith("profile_") and base.endswith(".json"):
                ids.add(base[len("profile_"):-len(".json")])

        # legacy files without extension (optional)
        for p in glob.glob(os.path.join(self.base_dir, "profile_*")):
            base = os.path.basename(p)
            if base.startswith("profile_") and not base.endswith(".json") and base != "profileManager.json":
                ids.add(base.split("profile_", 1)[1])

        return sorted(ids)


# ===================== COLLECTOR =====================

class CustomGestureCollector:
    def __init__(self, dataset_size=200, open_after=False):
        self.dataset_size = dataset_size
        self.open_after = open_after
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def open_folder(path):
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(f'explorer "{path}"')
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    @staticmethod
    def landmarks_to_feature_vector(hand_landmarks):
        xs, ys = [], []
        for lm in hand_landmarks.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        xs -= xs[0]
        ys -= ys[0]

        radii = np.sqrt(xs**2 + ys**2)
        max_r = np.max(radii)
        if max_r > 0:
            xs /= max_r
            ys /= max_r

        return np.concatenate([xs, ys], axis=0)  # (42,)

    def collect_gesture(self, gesture_name: str):
        folder = tempfile.mkdtemp(prefix=f"gesture_{gesture_name}_")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam.")
            shutil.rmtree(folder, ignore_errors=True)
            return None

        print(f"\n=== Gesture: {gesture_name} ===")
        print('Press "y" in the preview window to start recording. Press "q" to cancel.\n')

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                cv2.putText(frame, f'Gesture: {gesture_name}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, 'Press "y" to start, "q" to cancel', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Gesture Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    shutil.rmtree(folder, ignore_errors=True)
                    return None

            counter = 0
            while counter < self.dataset_size:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if not results.multi_hand_landmarks:
                    cv2.putText(frame, "Hold gesture in frame!", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Gesture Collector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                hand_lm = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                vec = self.landmarks_to_feature_vector(hand_lm)
                np.save(os.path.join(folder, f"{counter}.npy"), vec)

                cv2.putText(frame, f'Sample {counter+1}/{self.dataset_size}', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Gesture Collector", frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                counter += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"[OK] Recorded {counter} samples -> {folder}")
        if self.open_after:
            self.open_folder(folder)
        return folder


# ===================== DATASET MANAGER =====================

class CombinedDatasetManager:
    def __init__(self, paths: PathConfig):
        self.paths = paths

    def load_landmark_dataset(self):
        if not (os.path.exists(self.paths.X_path) and os.path.exists(self.paths.y_path) and os.path.exists(self.paths.classes_path)):
            print("[ERROR] Landmark dataset not found in:", self.paths.LANDMARK_DIR)
            return None, None, None
        X = np.load(self.paths.X_path)
        y = np.load(self.paths.y_path)
        class_names = np.load(self.paths.classes_path, allow_pickle=True)
        return X, y, class_names

    def save_landmark_dataset(self, X, y, class_names):
        np.save(self.paths.X_path, X)
        np.save(self.paths.y_path, y)
        np.save(self.paths.classes_path, class_names)
        print("[DATASET] Saved X/y/class_names.")

    def load_vectors_from_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            return None
        npy_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".npy"))
        if not npy_files:
            return None
        vecs = [np.load(os.path.join(folder_path, f)) for f in npy_files]
        return np.stack(vecs, axis=0)

    def add_new_gesture_from_folder(self, gesture_name: str, recording_folder: str) -> bool:
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False

        if gesture_name in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' already exists in dataset.")
            return False

        X_new = self.load_vectors_from_folder(recording_folder)
        if X_new is None:
            print("[ERROR] No vectors recorded.")
            return False

        new_label = len(class_names)
        y_new = np.full(X_new.shape[0], new_label, dtype=np.int64)

        X_out = np.concatenate([X, X_new], axis=0)
        y_out = np.concatenate([y, y_new], axis=0)
        class_out = np.append(class_names, gesture_name)

        self.save_landmark_dataset(X_out, y_out, class_out)
        print(f"[DATASET] Added '{gesture_name}' (class {new_label}).")
        return True

    def delete_gesture_from_landmarks(self, gesture_name: str) -> bool:
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False

        if gesture_name not in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' not found in dataset.")
            return False

        idx = int(np.where(class_names == gesture_name)[0][0])
        keep_mask = (y != idx)

        X_new = X[keep_mask]
        y_kept = y[keep_mask]
        class_new = np.delete(class_names, idx)

        kept_old_labels = sorted(set(y_kept.tolist()))
        old_to_new = {old: new for new, old in enumerate(kept_old_labels)}
        y_new = np.array([old_to_new[v] for v in y_kept], dtype=np.int64)

        self.save_landmark_dataset(X_new, y_new, class_new)
        print(f"[DATASET] Deleted '{gesture_name}'.")
        return True

    def replace_gesture_dataset_from_folder(self, gesture_name: str, recording_folder: str) -> bool:
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False
        if gesture_name not in class_names:
            print(f"[ERROR] Gesture '{gesture_name}' not found in dataset.")
            return False

        self.delete_gesture_from_landmarks(gesture_name)
        ok = self.add_new_gesture_from_folder(gesture_name, recording_folder)
        return ok

    def rename_gesture_in_dataset(self, old_name: str, new_name: str) -> bool:
        X, y, class_names = self.load_landmark_dataset()
        if X is None:
            return False
        if old_name not in class_names:
            print(f"[ERROR] '{old_name}' not found in dataset.")
            return False
        if new_name in class_names:
            print(f"[ERROR] '{new_name}' already exists in dataset.")
            return False

        idx = int(np.where(class_names == old_name)[0][0])
        class_names[idx] = new_name
        self.save_landmark_dataset(X, y, class_names)
        print(f"[DATASET] Renamed '{old_name}' -> '{new_name}'.")
        return True


# ===================== MENU =====================

def main():
    paths = PathConfig()
    profiles = ProfileFileStore(paths.BASE_DIR)
    dataset = CombinedDatasetManager(paths)
    collector = CustomGestureCollector(dataset_size=200, open_after=False)

    while True:
        print("\n===== Custom Gesture Manager =====")
        print("1) Create NEW gesture (dataset + write into profile_<id>.json)")
        print("2) Edit gesture (name/key/type/dataset)")
        print("3) Delete gesture (dataset + remove from ALL profiles)")
        print("4) Exit")
        choice = input("Select an option (1-4): ").strip()

        # -------- CREATE --------
        if choice == "1":
            gesture_name = input("Enter NEW gesture name (label): ").strip()
            if not gesture_name:
                continue

            key = input("Enter key_pressed (e.g. w, space, left): ").strip()
            raw_type = input("Enter input_type (click / hold): ").strip().lower()
            if raw_type not in ("click", "hold"):
                print("Invalid input_type.")
                continue
            input_type = raw_type.capitalize()

            profile_id = input("Enter profile ID to save into (e.g. 1): ").strip()
            if not profile_id:
                continue

            print("\nAbout to CREATE:")
            print("  gesture:", gesture_name)
            print("  key_pressed:", key)
            print("  input_type:", input_type)
            print("  target file:", f"profile_{profile_id}.json")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm != "YES":
                print("Cancelled.")
                continue

            folder = collector.collect_gesture(gesture_name)
            if folder is None:
                continue

            ok = dataset.add_new_gesture_from_folder(gesture_name, folder)
            shutil.rmtree(folder, ignore_errors=True)

            if ok:
                profiles.upsert_action(profile_id, gesture_name, key, input_type)
                print("[OK] Created + saved into JSON.")

        # -------- EDIT --------
        elif choice == "2":
            gesture_name = input("Enter existing gesture name to EDIT: ").strip()
            if not gesture_name:
                continue

            print("\nWhat do you want to edit?")
            print("1) Name (rename) [dataset + profiles]")
            print("2) key_pressed [profiles only]")
            print("3) input_type [profiles only]")
            print("4) Gesture dataset (re-record vectors) [dataset only]")
            sub = input("Select (1-4): ").strip()

            if sub == "1":
                new_name = input("Enter NEW name: ").strip()
                if not new_name:
                    continue
                confirm = input(f"Type 'YES' to rename '{gesture_name}' -> '{new_name}': ").strip()
                if confirm != "YES":
                    continue

                ok = dataset.rename_gesture_in_dataset(gesture_name, new_name)
                if ok:
                    for pid in profiles.list_profile_ids():
                        profiles.rename_action(pid, gesture_name, new_name)
                    print("[OK] Renamed in dataset + profiles.")

            elif sub == "2":
                new_key = input("Enter NEW key_pressed: ").strip()
                if new_key == "":
                    continue
                confirm = input("Type 'YES' to apply to ALL profiles containing this gesture: ").strip()
                if confirm != "YES":
                    continue

                for pid in profiles.list_profile_ids():
                    data = profiles.load_profile(pid)
                    if data is None:
                        continue
                    changed = False
                    for a in data.get("Actions", []):
                        if isinstance(a, dict) and a.get("name") == gesture_name:
                            a["key_pressed"] = new_key
                            changed = True
                    if changed:
                        profiles.save_profile(pid, data)

                print("[OK] Updated key_pressed.")

            elif sub == "3":
                new_type = input("Enter NEW input_type (Click/Hold): ").strip().capitalize()
                if new_type not in ("Click", "Hold"):
                    print("Invalid input_type.")
                    continue
                confirm = input("Type 'YES' to apply to ALL profiles containing this gesture: ").strip()
                if confirm != "YES":
                    continue

                for pid in profiles.list_profile_ids():
                    data = profiles.load_profile(pid)
                    if data is None:
                        continue
                    changed = False
                    for a in data.get("Actions", []):
                        if isinstance(a, dict) and a.get("name") == gesture_name:
                            a["input_type"] = new_type
                            changed = True
                    if changed:
                        profiles.save_profile(pid, data)

                print("[OK] Updated input_type.")

            elif sub == "4":
                confirm = input(f"Type 'YES' to re-record and REPLACE dataset vectors for '{gesture_name}': ").strip()
                if confirm != "YES":
                    continue

                folder = collector.collect_gesture(gesture_name)
                if folder is None:
                    continue

                ok = dataset.replace_gesture_dataset_from_folder(gesture_name, folder)
                shutil.rmtree(folder, ignore_errors=True)
                if ok:
                    print("[OK] Dataset replaced (profiles unchanged).")

            else:
                print("Invalid selection.")

        # -------- DELETE --------
        elif choice == "3":
            gesture_name = input("Enter gesture name to DELETE (dataset + all profiles): ").strip()
            if not gesture_name:
                continue

            print("\nWARNING: This will delete from:")
            print(" - landmark dataset (X/y/class_names)")
            print(" - ALL profile_*.json files")
            confirm = input("Type 'YES' to confirm delete: ").strip()
            if confirm != "YES":
                continue

            dataset_ok = dataset.delete_gesture_from_landmarks(gesture_name)

            removed_total = 0
            for pid in profiles.list_profile_ids():
                removed_total += profiles.remove_action(pid, gesture_name)

            print(f"[DONE] Delete complete. dataset_deleted={dataset_ok}, profile_actions_removed={removed_total}")

        elif choice == "4":
            print("Bye.")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
