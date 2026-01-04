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
import time


# ===================== PATH CONFIG =====================

class PathConfig:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.DATA_DIR = os.path.join(self.BASE_DIR, "data", "landmarkVectors")
        self.X_PATH = os.path.join(self.DATA_DIR, "X.npy")
        self.Y_PATH = os.path.join(self.DATA_DIR, "y.npy")
        self.CLASSES_PATH = os.path.join(self.DATA_DIR, "class_names.npy")

        os.makedirs(self.DATA_DIR, exist_ok=True)


# ===================== GESTURE LIST STORE (NEW LOGIC) =====================
# GestureList.json is the single source of truth for "available gestures".
# It stores a plain JSON list of strings, e.g. ["wave", "thumbs_up"].

class GestureListStore:
    def __init__(self, base_dir: str, filename: str = "GestureList.json"):
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, filename)

    def load(self) -> list[str]:
        if not os.path.exists(self.path):
            # Create an empty list on first run
            self.save([])
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("GestureList.json must be a JSON list of strings")
        # normalize: strings only, strip, unique, stable order
        out = []
        seen = set()
        for g in data:
            if not isinstance(g, str):
                continue
            g2 = g.strip()
            if not g2 or g2 in seen:
                continue
            out.append(g2)
            seen.add(g2)
        return out

    def save(self, gestures: list[str]) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(gestures, f, indent=4)

    def add(self, gesture: str) -> bool:
        gesture = (gesture or "").strip()
        if not gesture:
            return False
        gestures = self.load()
        if gesture in gestures:
            return False
        gestures.append(gesture)
        self.save(gestures)
        return True

    def remove(self, gesture: str) -> bool:
        gesture = (gesture or "").strip()
        gestures = self.load()
        if gesture not in gestures:
            return False
        gestures = [g for g in gestures if g != gesture]
        self.save(gestures)
        return True

    def rename(self, old: str, new: str) -> bool:
        old = (old or "").strip()
        new = (new or "").strip()
        if not old or not new or old == new:
            return False
        gestures = self.load()
        if old not in gestures:
            return False
        if new in gestures:
            return False
        gestures = [new if g == old else g for g in gestures]
        self.save(gestures)
        return True


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
        p_json = self.json_path(profile_id)
        p_legacy = self.legacy_path(profile_id)

        if os.path.exists(p_json):
            return p_json
        if os.path.exists(p_legacy):
            return p_legacy
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
            for a in data["Actions"]:
                if isinstance(a, dict) and "key_type" in a and isinstance(a["key_type"], str):
                    # normalize case
                    if a["key_type"].lower() == "keyboard":
                        a["key_type"] = "Keyboard"
                    elif a["key_type"].lower() == "mouse":
                        a["key_type"] = "Mouse"
            return data
        except Exception as e:
            print(f"[PROFILE] Failed to load profile_{profile_id}:", e)
            return None

    def save_profile(self, profile_id: str, data: dict):
        path = self.json_path(profile_id)
        data["Profile_ID"] = str(profile_id)
        if "Actions" not in data or not isinstance(data["Actions"], list):
            data["Actions"] = []
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def ensure_profile(self, profile_id: str) -> dict:
        data = self.load_profile(profile_id)
        if data is None:
            data = {"Profile_ID": str(profile_id), "Actions": []}
            self.save_profile(profile_id, data)
        return data

    # ===================== OLD PROFILE ACTION METHODS (COMMENTED OUT) =====================
    #     def upsert_action(self, profile_id: str, name: str, key_pressed, input_type, key_type):
    #         """
    #         Overwrite action if same 'name' exists, else append.
    #         """
    #         data = self.ensure_profile(profile_id)
    #         actions = data["Actions"]
    #
    #         # normalize type
    #         it = input_type
    #         if isinstance(it, str):
    #             it = it.capitalize()
    #             if it not in ("Click", "Hold"):
    #                 it = input_type
    #
    #         updated = False
    #         for a in actions:
    #             if isinstance(a, dict) and a.get("name") == name:
    #                 a["key_pressed"] = key_pressed
    #                 a["input_type"] = it
    #                 a["key_type"] = key_type
    #                 updated = True
    #
    #         if not updated:
    #             actions.append({
    #                 "name": name,
    #                 "key_pressed": key_pressed,
    #                 "input_type": it,
    #                 "key_type": key_type
    #             })
    #
    #         data["Actions"] = actions
    #         self.save_profile(profile_id, data)
    #         print(f"[PROFILE] {'Updated' if updated else 'Added'} action '{name}' in profile_{profile_id}.json")
    #
    #     def remove_action(self, profile_id: str, name: str):
    #         data = self.load_profile(profile_id)
    #         if not data:
    #             print(f"[PROFILE] profile_{profile_id} not found.")
    #             return
    #         actions = data.get("Actions", [])
    #         before = len(actions)
    #         actions = [a for a in actions if not (isinstance(a, dict) and a.get("name") == name)]
    #         after = len(actions)
    #         data["Actions"] = actions
    #         self.save_profile(profile_id, data)
    #         if after < before:
    #             print(f"[PROFILE] Removed action '{name}' from profile_{profile_id}.json")
    #         else:
    #             print(f"[PROFILE] No action '{name}' found.")
    #
    #     def rename_action(self, profile_id: str, old_name: str, new_name: str):
    #         data = self.load_profile(profile_id)
    #         if not data:
    #             print(f"[PROFILE] profile_{profile_id} not found.")
    #             return
    #         actions = data.get("Actions", [])
    #         changed = False
    #         for a in actions:
    #             if isinstance(a, dict) and a.get("name") == old_name:
    #                 a["name"] = new_name
    #                 changed = True
    #         data["Actions"] = actions
    #         self.save_profile(profile_id, data)
    #         print(f"[PROFILE] {'Renamed' if changed else 'No changes for'} '{old_name}' -> '{new_name}' in profile_{profile_id}.json")

    # ===================== PROFILE ACTION MAPPINGS (NEW LOGIC) =====================
    # NEW schema in profile_<id>.json:
    #   {
    #     "Profile_ID": "1",
    #     "Actions": [
    #        {"gesture": "thumbs_up", "key_pressed": "space", "input_type": "Click", "key_type": "Keyboard"},
    #        ...
    #     ]
    #   }
    #
    # Legacy compatibility:
    # - If an entry has "name" but no "gesture", we treat "name" as the gesture label.

    def _normalize_actions_new_schema(self, actions: list) -> list[dict]:
        out = []
        for a in actions or []:
            if not isinstance(a, dict):
                continue
            gesture = a.get("gesture")
            if not gesture:
                gesture = a.get("name")  # legacy fallback
            if not isinstance(gesture, str) or not gesture.strip():
                continue
            gesture = gesture.strip()

            key_pressed = a.get("key_pressed")
            input_type = a.get("input_type")
            key_type = a.get("key_type")

            # normalize types
            if isinstance(input_type, str):
                t = input_type.strip().lower()
                if t == "click":
                    input_type = "Click"
                elif t == "hold":
                    input_type = "Hold"
                elif t in ("d_click", "doubleclick", "double_click"):
                    input_type = "D_Click"

            if isinstance(key_type, str):
                kt = key_type.strip().lower()
                if kt == "mouse":
                    key_type = "Mouse"
                elif kt == "keyboard":
                    key_type = "Keyboard"

            out.append({
                "gesture": gesture,
                "key_pressed": key_pressed,
                "input_type": input_type,
                "key_type": key_type
            })
        return out

    def upsert_mapping(self, profile_id: str, gesture: str, key_pressed, input_type, key_type):
        """
        Add or update a mapping keyed by *gesture*.
        """
        gesture = (gesture or "").strip()
        if not gesture:
            raise ValueError("gesture cannot be empty")

        data = self.ensure_profile(profile_id)
        actions = self._normalize_actions_new_schema(data.get("Actions", []))

        # normalize input_type
        it = input_type
        if isinstance(it, str):
            t = it.strip().lower()
            if t == "click":
                it = "Click"
            elif t == "hold":
                it = "Hold"
            elif t in ("d_click", "doubleclick", "double_click"):
                it = "D_Click"

        # normalize key_type
        kt = key_type
        if isinstance(kt, str):
            t = kt.strip().lower()
            if t == "mouse":
                kt = "Mouse"
            elif t == "keyboard":
                kt = "Keyboard"

        updated = False
        for a in actions:
            if a.get("gesture") == gesture:
                a["key_pressed"] = key_pressed
                a["input_type"] = it
                a["key_type"] = kt
                updated = True
                break

        if not updated:
            actions.append({
                "gesture": gesture,
                "key_pressed": key_pressed,
                "input_type": it,
                "key_type": kt
            })

        data["Actions"] = actions
        self.save_profile(profile_id, data)
        print(f"[PROFILE] {'Updated' if updated else 'Added'} mapping for gesture '{gesture}' in profile_{profile_id}.json")

    def remove_mapping(self, profile_id: str, gesture: str):
        gesture = (gesture or "").strip()
        data = self.load_profile(profile_id)
        if not data:
            print(f"[PROFILE] profile_{profile_id} not found.")
            return

        actions = self._normalize_actions_new_schema(data.get("Actions", []))
        before = len(actions)
        actions = [a for a in actions if a.get("gesture") != gesture]
        after = len(actions)

        data["Actions"] = actions
        self.save_profile(profile_id, data)
        if after < before:
            print(f"[PROFILE] Removed mapping for gesture '{gesture}' from profile_{profile_id}.json")
        else:
            print(f"[PROFILE] No mapping found for gesture '{gesture}'")

    def rename_mapping_gesture(self, profile_id: str, old_gesture: str, new_gesture: str):
        old_gesture = (old_gesture or "").strip()
        new_gesture = (new_gesture or "").strip()
        data = self.load_profile(profile_id)
        if not data:
            print(f"[PROFILE] profile_{profile_id} not found.")
            return

        actions = self._normalize_actions_new_schema(data.get("Actions", []))
        # avoid duplicates
        if any(a.get("gesture") == new_gesture for a in actions):
            print(f"[PROFILE] Gesture '{new_gesture}' already exists in profile_{profile_id}.json")
            return

        changed = False
        for a in actions:
            if a.get("gesture") == old_gesture:
                a["gesture"] = new_gesture
                changed = True

        data["Actions"] = actions
        self.save_profile(profile_id, data)
        print(f"[PROFILE] {'Renamed' if changed else 'No changes for'} '{old_gesture}' -> '{new_gesture}' in profile_{profile_id}.json")

    def list_profile_ids(self) -> list[str]:
        out = []
        for p in glob.glob(os.path.join(self.base_dir, "profile_*.json")):
            base = os.path.basename(p)
            # profile_1.json -> 1
            try:
                pid = base.replace("profile_", "").replace(".json", "")
                out.append(pid)
            except Exception:
                pass
        # stable order, numeric first if possible
        def key_fn(x):
            return (0, int(x)) if x.isdigit() else (1, x)
        return sorted(set(out), key=key_fn)


# ===================== VIDEO + COLLECTION =====================

class CustomGestureCollector:
    def __init__(self, dataset_size=200, open_after=False):
        self.dataset_size = dataset_size
        self.open_after = open_after

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def open_folder(self, path):
        try:
            system = platform.system().lower()
            if "windows" in system:
                os.startfile(path)
            elif "darwin" in system:
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print("Could not open folder:", e)

    def landmarks_to_feature_vector(self, lm):
        coords = []
        for p in lm.landmark:
            coords.append([p.x, p.y])
        coords = np.array(coords, dtype=np.float32)
        wrist = coords[0].copy()
        coords = coords - wrist
        scale = np.linalg.norm(coords[9]) + 1e-6
        coords = coords / scale
        return coords.reshape(-1)

    def collect_gesture(self, gesture_name: str, output_folder: str,
                        capture_interval: float = 0.08,
                        require_hand: bool = True) -> bool:
        """
        AUTO-CAPTURE (ARMED BY SPACE):

        - Starts paused (NOT capturing).
        - SPACE toggles capture ON/OFF (pause/unpause).
        - Q cancels:
            * deletes any saved samples in output_folder
            * returns False (so caller does NOT save dataset / gesturelist)
        - Saves one sample every capture_interval seconds while capturing AND hand detected.

        Returns True only if dataset_size samples were collected.
        """

        import time  # safe even if already imported

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CAM] Failed to open camera.")
            return False

        os.makedirs(output_folder, exist_ok=True)
        saved = 0

        capturing = False  # start paused (armed by SPACE)
        last_save_t = time.perf_counter()

        print("\n[COLLECT] AUTO-CAPTURE (SPACE to start/pause)")
        print("  SPACE -> start / pause capture")
        print("  Q     -> cancel (saves NOTHING)\n")
        print(f"[COLLECT] Target: {self.dataset_size} samples | Interval: {capture_interval:.3f}s\n")

        def cleanup_folder():
            try:
                for fn in os.listdir(output_folder):
                    if fn.lower().endswith(".npy"):
                        try:
                            os.remove(os.path.join(output_folder, fn))
                        except Exception:
                            pass
            except Exception:
                pass

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(frame_rgb)

            hand_ok = res.multi_hand_landmarks is not None

            # overlay text
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {saved}/{self.dataset_size}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {'YES' if hand_ok else 'NO'}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Capturing: {'ON' if capturing else 'OFF'} (SPACE)", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(frame, "Q = Cancel (saves NOTHING)", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # draw landmarks
            if hand_ok:
                for hand_lm in res.multi_hand_landmarks:
                    for p in hand_lm.landmark:
                        h, w = frame.shape[:2]
                        cx, cy = int(p.x * w), int(p.y * h)
                        cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

            cv2.imshow("Collect Gesture Samples (AUTO)", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit/cancel -> delete everything and return False
            if key == ord('q'):
                cleanup_folder()
                cap.release()
                cv2.destroyAllWindows()
                return False

            # SPACE toggles capture ON/OFF
            if key == 32:  # space
                capturing = not capturing
                # reset timer so it doesn't instantly save 10 files in a burst
                last_save_t = time.perf_counter()

            # If finished, exit successfully
            if saved >= self.dataset_size:
                cap.release()
                cv2.destroyAllWindows()
                if self.open_after:
                    self.open_folder(output_folder)
                return True

            if not capturing:
                continue

            # only save at interval
            now = time.perf_counter()
            if now - last_save_t < capture_interval:
                continue

            if require_hand and not hand_ok:
                continue

            if hand_ok:
                hand_lm = res.multi_hand_landmarks[0]
                vec = self.landmarks_to_feature_vector(hand_lm)
                npy_path = os.path.join(output_folder, f"{gesture_name}_{saved:05d}.npy")
                np.save(npy_path, vec)
                saved += 1
                last_save_t = now

            else:
                # if require_hand=False, you could save blank/noise, but generally not desired
                last_save_t = now

        cap.release()
        cv2.destroyAllWindows()

        if self.open_after:
            self.open_folder(output_folder)

        return True


# ===================== DATASET OPS (.npy) =====================

class CombinedDatasetManager:
    def __init__(self, paths: PathConfig):
        self.paths = paths

    def load_landmark_dataset(self):
        if not os.path.exists(self.paths.X_PATH) or not os.path.exists(self.paths.Y_PATH) or not os.path.exists(self.paths.CLASSES_PATH):
            return None, None, []
        X = np.load(self.paths.X_PATH, allow_pickle=True)
        y = np.load(self.paths.Y_PATH, allow_pickle=True)
        classes = np.load(self.paths.CLASSES_PATH, allow_pickle=True).tolist()
        return X, y, classes

    def save_landmark_dataset(self, X, y, classes):
        np.save(self.paths.X_PATH, X)
        np.save(self.paths.Y_PATH, y)
        np.save(self.paths.CLASSES_PATH, np.array(classes, dtype=object))

    def load_vectors_from_folder(self, folder: str):
        vecs = []
        for p in sorted(glob.glob(os.path.join(folder, "*.npy"))):
            try:
                v = np.load(p, allow_pickle=True)
                vecs.append(v)
            except Exception:
                pass
        if not vecs:
            return None
        return np.stack(vecs, axis=0)

    def add_new_gesture_from_folder(self, gesture_name: str, folder: str):
        X, y, classes = self.load_landmark_dataset()
        new_vecs = self.load_vectors_from_folder(folder)
        if new_vecs is None:
            raise RuntimeError("No .npy vectors found in folder")

        if X is None:
            X = new_vecs
            y = np.array([gesture_name] * len(new_vecs), dtype=object)
            classes = [gesture_name]
        else:
            X = np.concatenate([X, new_vecs], axis=0)
            y = np.concatenate([y, np.array([gesture_name] * len(new_vecs), dtype=object)], axis=0)
            if gesture_name not in classes:
                classes.append(gesture_name)

        self.save_landmark_dataset(X, y, classes)

    def delete_gesture_from_landmarks(self, gesture_name: str):
        X, y, classes = self.load_landmark_dataset()
        if X is None:
            print("[DATASET] No dataset found.")
            return

        mask = np.array([lbl != gesture_name for lbl in y], dtype=bool)
        X2 = X[mask]
        y2 = y[mask]
        classes2 = [c for c in classes if c != gesture_name]
        self.save_landmark_dataset(X2, y2, classes2)

    def replace_gesture_dataset_from_folder(self, gesture_name: str, folder: str):
        self.delete_gesture_from_landmarks(gesture_name)
        self.add_new_gesture_from_folder(gesture_name, folder)

    def rename_gesture_in_dataset(self, old_name: str, new_name: str):
        X, y, classes = self.load_landmark_dataset()
        if X is None:
            print("[DATASET] No dataset found.")
            return

        y2 = np.array([new_name if lbl == old_name else lbl for lbl in y], dtype=object)
        classes2 = [new_name if c == old_name else c for c in classes]
        if new_name not in classes2:
            classes2.append(new_name)

        self.save_landmark_dataset(X, y2, classes2)


# ===================== OLD MAIN (COMMENTED OUT) =====================
# def main():
#     paths = PathConfig()
#     profiles = ProfileFileStore(paths.BASE_DIR)
#     dataset = CombinedDatasetManager(paths)
#     collector = CustomGestureCollector(dataset_size=200, open_after=False)
#
#     while True:
#         print("\n===== Custom Gesture Manager =====")
#         print("1) Create NEW gesture (dataset + write into profile_<id>.json)")
#         print("2) Edit gesture (name/key/type/dataset)")
#         print("3) Delete gesture (dataset + remove from ALL profiles)")
#         print("4) Exit")
#         choice = input("Select an option (1-4): ").strip()
#
#         # -------- CREATE --------
#         if choice == "1":
#             gesture_name = input("Enter NEW gesture name (label): ").strip()
#             if not gesture_name:
#                 continue
#
#             key = input("Enter key_pressed (e.g. w, space, left): ").strip()
#             raw_type = input("Enter input_type (click / hold / D_Click): ").strip().lower()
#             if raw_type not in ("click", "hold" , "D_Click"):
#                 print("Invalid input_type.")
#                 continue
#             input_type = raw_type.capitalize()
#
#             raw_key_type = input("Enter key_type (keyboard / mouse): ").strip().lower()
#             if raw_key_type not in ("keyboard", "mouse"):
#                 print("Invalid key_type.")
#                 continue
#             key_type = raw_key_type.capitalize()
#
#             profile_id = input("Enter profile_id to write into (e.g. 1): ").strip()
#             if not profile_id:
#                 profile_id = "1"
#
#             out_folder = tempfile.mkdtemp(prefix=f"{gesture_name}_")
#             print(f"[DATASET] Collecting samples for '{gesture_name}' into: {out_folder}")
#             ok = collector.collect_gesture(gesture_name, out_folder)
#             if ok:
#                 dataset.add_new_gesture_from_folder(gesture_name, out_folder)
#                 profiles.upsert_action(profile_id, gesture_name, key, input_type, key_type)
#                 print(f"[DONE] Created gesture '{gesture_name}' + saved to profile_{profile_id}.json")
#             else:
#                 print("[DATASET] Collection cancelled/skipped.")
#             shutil.rmtree(out_folder, ignore_errors=True)
#
#         # -------- EDIT --------
#         elif choice == "2":
#             old_name = input("Enter existing gesture name to edit: ").strip()
#             if not old_name:
#                 continue
#             new_name = input("Enter NEW name (or blank to keep): ").strip()
#             key = input("Enter NEW key_pressed (or blank to keep): ").strip()
#             raw_type = input("Enter NEW input_type (click/hold/D_Click) (or blank): ").strip().lower()
#             raw_key_type = input("Enter NEW key_type (keyboard/mouse) (or blank): ").strip().lower()
#
#             profile_id = input("Enter profile_id to update (e.g. 1): ").strip()
#             if not profile_id:
#                 profile_id = "1"
#
#             # rename dataset label if requested
#             if new_name:
#                 dataset.rename_gesture_in_dataset(old_name, new_name)
#                 profiles.rename_action(profile_id, old_name, new_name)
#                 old_name = new_name
#
#             # update action mapping if key/type updated
#             if key or raw_type or raw_key_type:
#                 data = profiles.load_profile(profile_id) or {"Profile_ID": profile_id, "Actions": []}
#                 actions = data.get("Actions", [])
#                 for a in actions:
#                     if isinstance(a, dict) and a.get("name") == old_name:
#                         if key:
#                             a["key_pressed"] = key
#                         if raw_type:
#                             a["input_type"] = raw_type.capitalize()
#                         if raw_key_type:
#                             a["key_type"] = raw_key_type.capitalize()
#                 data["Actions"] = actions
#                 profiles.save_profile(profile_id, data)
#                 print(f"[PROFILE] Updated '{old_name}' in profile_{profile_id}.json")
#
#         # -------- DELETE --------
#         elif choice == "3":
#             gesture_name = input("Enter gesture name to DELETE: ").strip()
#             if not gesture_name:
#                 continue
#             dataset.delete_gesture_from_landmarks(gesture_name)
#             # remove from ALL profiles
#             for pid in profiles.list_profile_ids():
#                 profiles.remove_action(pid, gesture_name)
#             print(f"[DONE] Deleted gesture '{gesture_name}'")
#
#         elif choice == "4":
#             print("Bye.")
#             break
#
#         else:
#             print("Invalid option.")
#
#
# if __name__ == "__main__":
#     main()

# ===================== NEW MAIN (ACTIVE) =====================

def main():
    paths = PathConfig()
    profiles = ProfileFileStore(paths.BASE_DIR)
    gesture_store = GestureListStore(paths.BASE_DIR)  # GestureList.json lives in BASE_DIR
    dataset = CombinedDatasetManager(paths)
    collector = CustomGestureCollector(dataset_size=200, open_after=False)

    def choose_profile_id() -> str:
        # Try to suggest existing profiles, but still allow arbitrary ID.
        try:
            existing = profiles.list_profile_ids()
        except Exception:
            existing = []
        if existing:
            print(f"Existing profiles: {', '.join(existing)}")
        pid = input("Enter profile id (e.g. 1): ").strip()
        return pid if pid else "1"

    while True:
        print("\n===== Custom Gesture Manager (NEW LOGIC) =====")
        print("1) Create NEW gesture (dataset + add to GestureList.json)")
        print("2) Rename gesture (dataset + GestureList.json + update profiles)")
        print("3) Delete gesture (dataset + GestureList.json + remove from ALL profiles)")
        print("4) Add/Update profile mapping (choose gesture from GestureList)")
        print("5) Remove profile mapping (by gesture)")
        print("6) Exit")
        choice = input("Select an option (1-6): ").strip()

        # -------- 1) CREATE gesture label (dataset + GestureList) --------
        if choice == "1":
            gesture_name = input("Enter NEW gesture name (label): ").strip()
            if not gesture_name:
                continue

            # IMPORTANT: do NOT add to GestureList yet.
            # If user cancels (Q), nothing should be saved anywhere.
            existing_gestures = gesture_store.load()
            already_exists = gesture_name in existing_gestures

            out_folder = None
            try:
                out_folder = tempfile.mkdtemp(prefix=f"{gesture_name}_")
                print(f"[DATASET] Collecting samples for '{gesture_name}' into: {out_folder}")

                ok = collector.collect_gesture(gesture_name, out_folder)  # SPACE to start/pause, Q cancels

                if not ok:
                    print("[CANCELLED] Nothing saved. Returning to menu.")
                    # Temp folder will be removed below.
                    continue

                # SUCCESS -> now we save for real
                if not already_exists:
                    gesture_store.add(gesture_name)
                    print(f"[GestureList] Added '{gesture_name}'")
                else:
                    print(f"[GestureList] '{gesture_name}' already exists (kept)")

                dataset.add_new_gesture_from_folder(gesture_name, out_folder)
                print(f"[DATASET] Added '{gesture_name}' to landmarkVectors dataset.")

            finally:
                if out_folder:
                    shutil.rmtree(out_folder, ignore_errors=True)


        # -------- 2) RENAME gesture label everywhere --------
        elif choice == "2":
            gestures = gesture_store.load()
            if gestures:
                print("Gestures:", ", ".join(gestures))
            old_name = input("Enter existing gesture name to rename: ").strip()
            new_name = input("Enter NEW gesture name: ").strip()
            if not old_name or not new_name:
                continue

            # GestureList rename
            if gesture_store.rename(old_name, new_name):
                print(f"[GestureList] Renamed '{old_name}' -> '{new_name}'")
            else:
                print("[GestureList] Rename failed (old not found, or new already exists).")

            # Dataset rename
            try:
                dataset.rename_gesture_in_dataset(old_name, new_name)
                print(f"[DATASET] Renamed '{old_name}' -> '{new_name}' in landmarkVectors dataset.")
            except Exception as e:
                print("[DATASET] Rename failed:", e)

            # Update ALL profiles: mapping gesture key changes
            for pid in profiles.list_profile_ids():
                try:
                    profiles.rename_mapping_gesture(pid, old_name, new_name)
                except Exception as e:
                    print(f"[PROFILE] Failed to update profile_{pid}.json:", e)

        # -------- 3) DELETE gesture label everywhere --------
        elif choice == "3":
            gestures = gesture_store.load()
            if gestures:
                print("Gestures:", ", ".join(gestures))
            gesture_name = input("Enter gesture name to DELETE: ").strip()
            if not gesture_name:
                continue

            # GestureList remove
            removed = gesture_store.remove(gesture_name)
            if removed:
                print(f"[GestureList] Removed '{gesture_name}'")
            else:
                print(f"[GestureList] '{gesture_name}' was not in GestureList.json")

            # Dataset delete
            try:
                dataset.delete_gesture_from_landmarks(gesture_name)
                print(f"[DATASET] Removed '{gesture_name}' from landmarkVectors dataset.")
            except Exception as e:
                print("[DATASET] Delete failed:", e)

            # Remove from ALL profiles
            for pid in profiles.list_profile_ids():
                try:
                    profiles.remove_mapping(pid, gesture_name)
                except Exception as e:
                    print(f"[PROFILE] Failed to update profile_{pid}.json:", e)

        # -------- 4) ADD / UPDATE profile mapping --------
        elif choice == "4":
            pid = choose_profile_id()

            gestures = gesture_store.load()
            if not gestures:
                print("[GestureList] No gestures available. Add some first.")
                continue

            print("\nAvailable gestures:")
            for i, g in enumerate(gestures, 1):
                print(f"  {i}) {g}")

            sel = input("Choose gesture by number (or type name): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(gestures):
                gesture = gestures[int(sel) - 1]
            else:
                gesture = sel

            if gesture not in gestures:
                print("[GestureList] Invalid gesture (not in GestureList.json).")
                continue

            key_pressed = input("Enter key_pressed (e.g. w, space, left): ").strip()
            raw_type = input("Enter input_type (click / hold): ").strip().lower()
            if raw_type not in ("click", "hold"):
                print("Invalid input_type.")
                continue
            input_type = "Click" if raw_type == "click" else "Hold"

            raw_key_type = input("Enter key_type (keyboard / mouse): ").strip().lower()
            if raw_key_type not in ("keyboard", "mouse"):
                print("Invalid key_type.")
                continue
            key_type = "Keyboard" if raw_key_type == "keyboard" else "Mouse"

            profiles.upsert_mapping(pid, gesture, key_pressed, input_type, key_type)

        # -------- 5) REMOVE profile mapping --------
        elif choice == "5":
            pid = choose_profile_id()
            gestures = gesture_store.load()
            if gestures:
                print("Gestures:", ", ".join(gestures))
            gesture = input("Enter gesture to unbind from this profile: ").strip()
            if not gesture:
                continue
            profiles.remove_mapping(pid, gesture)

        # -------- 6) EXIT --------
        elif choice == "6":
            print("Bye.")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
