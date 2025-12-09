import os
import time
import numpy as np
import cv2
import pyautogui
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp

from ProfileManager import ProfileManager
from Profiles import Profile
from Actions import Actions

# ================== PATH CONFIG ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")  # landmark dataset folder

# ================== CONSTANTS ====================

K_NEIGHBORS = 3
SMOOTH_WINDOW = 3
GESTURE_CONF_THRESHOLD = 0.6  # min probability to trust a gesture


# =================================================
#   HAND FEATURE EXTRACTOR (LANDMARKS -> 42-D VECTORS)
# =================================================

class HandFeatureExtractor:
    """Converts MediaPipe hand landmarks into normalized 42-D feature vectors."""

    @staticmethod
    def landmarks_to_vec_pair(hand_lms):
        """
        Convert MediaPipe hand landmarks to TWO normalized 42-D vectors:
          - vec_orig: original orientation
          - vec_mirror: horizontally mirrored around the wrist (xs -> -xs)
        """
        xs = []
        ys = []
        for lm in hand_lms.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        # Translate so wrist is origin
        xs -= xs[0]
        ys -= ys[0]

        # Scale by max distance from wrist
        radii = np.sqrt(xs**2 + ys**2)
        max_r = np.max(radii)
        if max_r > 0:
            xs_norm = xs / max_r
            ys_norm = ys / max_r
        else:
            xs_norm = xs
            ys_norm = ys

        # Original orientation
        vec_orig = np.concatenate([xs_norm, ys_norm], axis=0)  # (42,)

        # Mirrored horizontally (x -> -x)
        xs_mirror = -xs_norm
        vec_mirror = np.concatenate([xs_mirror, ys_norm], axis=0)

        return vec_orig, vec_mirror


# =================================================
#   KNN GESTURE MODEL
# =================================================

class GestureKNNModel:
    """
    Wraps the KNN classifier trained on landmark vectors.
    Provides classify(hand_lms) -> (label, confidence).
    """

    def __init__(self, data_dir, k_neighbors=3):
        self.data_dir = data_dir
        self.k_neighbors = k_neighbors
        self.knn = None
        self.class_names = None
        self.id_to_name = None
        self._load_and_train()

    def _load_and_train(self):
        X_path = os.path.join(self.data_dir, "X.npy")
        y_path = os.path.join(self.data_dir, "y.npy")
        classes_path = os.path.join(self.data_dir, "class_names.npy")

        if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(classes_path)):
            raise FileNotFoundError(
                f"Could not find X.npy / y.npy / class_names.npy in {self.data_dir}"
            )

        X = np.load(X_path)  # shape (N, 42)
        y = np.load(y_path)
        self.class_names = np.load(classes_path, allow_pickle=True)

        print("[MODEL] Loaded X:", X.shape)
        print("[MODEL] Loaded y:", y.shape)
        print("[MODEL] Classes:", self.class_names)

        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn.fit(X, y)

        self.id_to_name = {i: name for i, name in enumerate(self.class_names)}
        print("[MODEL] k-NN trained on landmark vectors.")

    def classify_landmarks(self, hand_lms):
        """
        Classify a single hand based on landmarks.
        Returns (label, confidence) where:
          - label is a string from class_names
          - confidence is max probability from k-NN
        Uses both original and mirrored vectors, picks the more confident.
        """
        vec_orig, vec_mirror = HandFeatureExtractor.landmarks_to_vec_pair(hand_lms)

        best_label = "none"
        best_conf = -1.0

        for vec in (vec_orig, vec_mirror):
            proba = self.knn.predict_proba([vec])[0]  # (num_classes,)
            top_idx = int(np.argmax(proba))
            top_conf = float(proba[top_idx])
            if top_conf > best_conf:
                best_conf = top_conf
                best_label = self.id_to_name[top_idx]

        return best_label, best_conf


# =================================================
#   TKINTER GUI FOR DEBUG STATUS
# =================================================

class ClickTesterGUI:
    """
    Small Tkinter window:
      - big button showing click count (increments on real left click)
      - label showing last smoothed gesture
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Controller Tester")
        self.root.geometry("380x260")

        self.click_count = 0
        self.click_var = tk.StringVar(value="Clicks: 0")
        self.status_var = tk.StringVar(value="Last gesture: none")

        self._build_widgets()

    def _build_widgets(self):
        btn = tk.Button(
            self.root,
            textvariable=self.click_var,
            font=("Arial", 20),
            width=10,
            height=2,
            command=self.increment_click_counter,
        )
        btn.pack(expand=True, fill="both", pady=10)

        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 12),
        )
        status_label.pack(pady=5)

    def increment_click_counter(self):
        self.click_count += 1
        self.click_var.set(f"Clicks: {self.click_count}")

    def set_last_gesture(self, gesture_name):
        self.status_var.set(f"Last gesture: {gesture_name}")

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# =================================================
#   GESTURE CONTROLLER APP (MAIN LOOP) + PROFILEMANAGER
# =================================================

class GestureControllerApp:
    """
    Main application:
      - captures webcam
      - uses MediaPipe to detect hands
      - uses GestureKNNModel to classify:
          * 'point'     -> pointer hand (moves mouse)
          * any other class -> treated as gesture ID
      - pointer and action hands MUST be different hands (when 2 hands)
      - sends the **smoothed gesture name** to Profile.callfunction(...)
    """

    def __init__(self, data_dir, active_profile_name="1"):
        self.data_dir = data_dir

        # Model
        self.model = GestureKNNModel(data_dir=self.data_dir, k_neighbors=K_NEIGHBORS)

        # GUI
        self.gui = ClickTesterGUI()

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Smoothing buffer
        self.last_action_predictions = []  # gesture names (e.g., "left_click", "hold", "w", etc.)
        self.last_smoothed_action = "none"

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Cannot open webcam.")

        self.screen_w, self.screen_h = pyautogui.size()
        print("[APP] Screen size:", self.screen_w, "x", self.screen_h)
        print("[APP] Press 'q' in the OpenCV window to quit.")

        # ---- ProfileManager setup ----
        self.profile_manager = self._load_or_create_manager()
        self.active_profile_name = active_profile_name
        self.active_profile = self._get_or_create_profile(active_profile_name)

    # ---------- PROFILE MANAGER HELPERS ----------

    def _load_or_create_manager(self):
        """
        Load ProfileManager from JSON if it exists,
        otherwise create an empty manager and save it.
        """
        pm_file = "profileManager.json"
        if os.path.exists(pm_file):
            print("[PROFILE] Loading ProfileManager from profileManager.json")
            return ProfileManager.readFile(pm_file)
        else:
            print("[PROFILE] profileManager.json not found, creating new empty manager")
            mgr = ProfileManager([])
            mgr.writeFile(pm_file)
            return mgr

    def _get_or_create_profile(self, profile_name):
        """
        Get an existing profile from the manager.
        If it does not exist, create a new empty one via manager.addProfile().
        """
        profile = self.profile_manager.getProfile(profile_name)
        if profile is not None:
            print(f"[PROFILE] Loaded existing profile '{profile_name}'")
            return profile

        print(f"[PROFILE] Profile '{profile_name}' not found, creating new one.")
        self.profile_manager.addProfile(profile_name)
        return self.profile_manager.getProfile(profile_name)

    # ---------- HAND INFO ----------

    def _get_hands_info(self, results, frame_w, frame_h):
        infos = []
        if not (results.multi_hand_landmarks and results.multi_handedness):
            return infos

        for hand_lms, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            mp_label = handedness.classification[0].label  # "Left" / "Right"

            tip = hand_lms.landmark[8]
            fx = int(tip.x * frame_w)
            fy = int(tip.y * frame_h)

            xs = [lm.x * frame_w for lm in hand_lms.landmark]
            center_x = float(np.mean(xs))

            infos.append(
                {
                    "hand_lms": hand_lms,
                    "mp_label": mp_label,
                    "center_x": center_x,
                    "tip_px": (fx, fy),
                    "tip_norm": (tip.x, tip.y),
                    "raw_label": "none",
                    "raw_conf": 0.0,
                }
            )

        return infos

    # ---------- MAIN LOOP ----------

    def run(self):
        try:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            ) as hands:
                while True:
                    # keep GUI responsive
                    self.gui.update()

                    ret, frame = self.cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    # ----- Detect and classify each hand -----
                    hands_info = self._get_hands_info(results, w, h)

                    for info in hands_info:
                        raw_label, raw_conf = self.model.classify_landmarks(
                            info["hand_lms"]
                        )
                        info["raw_label"] = raw_label
                        info["raw_conf"] = raw_conf

                    # ----- Pointer hand selection (prefer 'point') -----
                    pointer_hand = None
                    if hands_info:
                        point_candidates = [
                            hi for hi in hands_info
                            if hi["raw_label"] == "point"
                            and hi["raw_conf"] >= GESTURE_CONF_THRESHOLD
                        ]
                        if point_candidates:
                            pointer_hand = max(
                                point_candidates, key=lambda hi: hi["raw_conf"]
                            )
                        else:
                            pointer_hand = hands_info[0]

                    # Move mouse with pointer hand
                    pointer_debug = "None"
                    if pointer_hand is not None:
                        fx, fy = pointer_hand["tip_px"]
                        nx, ny = pointer_hand["tip_norm"]

                        # draw fingertip for pointer hand
                        cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

                        sx = int(nx * self.screen_w)
                        sy = int(ny * self.screen_h)
                        pyautogui.moveTo(sx, sy, duration=0)

                        pointer_debug = (
                            f"{pointer_hand['mp_label']} "
                            f"({pointer_hand['raw_label']}, {pointer_hand['raw_conf']:.2f})"
                        )

                    # ----- Action hand selection (must be different from pointer if 2 hands) -----
                    action_hand = None
                    action_label = "none"
                    action_conf = 0.0
                    action_debug = "None"

                    if hands_info:
                        if len(hands_info) >= 2:
                            for hi in hands_info:
                                if hi is pointer_hand:
                                    continue  # skip pointer

                                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                                    continue

                                # 'point' is reserved for pointer only
                                if hi["raw_label"] == "point":
                                    continue

                                if hi["raw_conf"] > action_conf:
                                    action_conf = hi["raw_conf"]
                                    action_label = hi["raw_label"]
                                    action_hand = hi

                    if action_hand is not None:
                        action_debug = (
                            f"{action_hand['mp_label']} "
                            f"({action_label}, {action_conf:.2f})"
                        )
                        self.mp_drawing.draw_landmarks(
                            frame,
                            action_hand["hand_lms"],
                            self.mp_hands.HAND_CONNECTIONS,
                        )
                    else:
                        action_label = "none"
                        action_conf = 0.0

                    # ----- Smooth gesture name over last N frames -----
                    # We smooth **gesture names** directly (e.g. "left_click", "hold", "w", "Jump")
                    self.last_action_predictions.append(action_label)
                    if len(self.last_action_predictions) > SMOOTH_WINDOW:
                        self.last_action_predictions.pop(0)

                    # Majority vote
                    counts = {}
                    for g in self.last_action_predictions:
                        counts[g] = counts.get(g, 0) + 1

                    smoothed_action = max(counts, key=counts.get)
                    self.last_smoothed_action = smoothed_action

                    # ===== ProfileManager CALL =====
                    # smoothed_action is the gesture name we pass to the profile.
                    # It should match an Actions.getName() inside the active profile.
                    if smoothed_action != "none":
                        action_obj = self.active_profile.getAction(smoothed_action)

                        if action_obj is not None:
                            # --- STOP holds for all other actions in the profile ---
                            for a in self.active_profile.getActionList():
                                if a != action_obj and a.getInputType() == "Hold":
                                    a.stopHold()

                            # --- USE the current action ---
                            action_obj.useAction(smoothed_action)
                        else:
                            # Optional debug:
                            # print(f"[PROFILE] No action mapped for gesture '{smoothed_action}' in profile {self.active_profile.getProfileID()}")
                            pass

                    # Update GUI with last gesture
                    self.gui.set_last_gesture(smoothed_action)

                    # ----- Debug overlay -----
                    debug_text = (
                        f"Pointer: {pointer_debug} | "
                        f"Action: {action_debug} | "
                        f"Smoothed: {smoothed_action}"
                    )
                    cv2.putText(
                        frame,
                        debug_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("Gesture Pointer Prototype (OOP + ProfileManager)", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    time.sleep(0.01)

        finally:
            self._cleanup()

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.gui.destroy()


# =================================================
#   MAIN ENTRY POINT
# =================================================

if __name__ == "__main__":
    # Change "1" to whichever profile ID you want as default
    app = GestureControllerApp(DATA_DIR, active_profile_name="1")
    app.run()
