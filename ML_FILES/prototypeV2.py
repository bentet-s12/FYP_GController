import os
import time
import numpy as np
import cv2
import pyautogui
pyautogui.PAUSE = 0
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import pydirectinput
import ctypes

from ProfileManager import ProfileManager
from Profiles import Profile
from Actions import Actions

# ================== PATH CONFIG ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")  # landmark dataset folder

# ================== CONSTANTS ====================

K_NEIGHBORS = 3
GESTURE_CONF_THRESHOLD = 0.6  # min probability to trust a gesture
pyautogui.PAUSE = 0

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

        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors, n_jobs=-1)
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
    Tkinter window:
      - big button showing click count
      - text showing last gesture
      - text showing current control mode
      - buttons to switch modes
    """

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Gesture Controller Tester")
        self.root.geometry("380x300")

        self.click_count = 0
        self.click_var = tk.StringVar(value="Clicks: 0")
        self.status_var = tk.StringVar(value="Last gesture: none")

        # NEW — shows current mode in the GUI
        self.mode_display_var = tk.StringVar(value=f"Current Mode: {self.app.control_mode.capitalize()}")

        self._build_widgets()

    def _build_widgets(self):
        # Click counter button
        btn = tk.Button(
            self.root,
            textvariable=self.click_var,
            font=("Arial", 20),
            width=10,
            height=2,
            command=self.increment_click_counter,
        )
        btn.pack(expand=True, fill="both", pady=10)

        # Gesture status text
        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 12),
        )
        status_label.pack(pady=3)

        # 🔥 NEW: Mode text display
        mode_label = tk.Label(
            self.root,
            textvariable=self.mode_display_var,
            font=("Arial", 12, "bold"),
            fg="blue"
        )
        mode_label.pack(pady=3)

        # Handedness buttons
        hand_frame = tk.Frame(self.root)
        hand_frame.pack(side="bottom", fill="x", pady=10)

        tk.Button(
            hand_frame,
            text="Left-handed",
            command=lambda: self.set_mode("left")
        ).pack(side="left", expand=True, padx=10)

        tk.Button(
            hand_frame,
            text="Right-handed",
            command=lambda: self.set_mode("right")
        ).pack(side="right", expand=True, padx=10)

        tk.Button(
            hand_frame,
            text="Auto",
            command=lambda: self.set_mode("auto")
        ).pack(side="bottom", expand=True, pady=5)

    def set_mode(self, mode: str):
        """Called when GUI buttons are pressed."""
        mode = mode.lower()
        self.app.control_mode = mode  # update app
        self.mode_display_var.set(f"Current Mode: {mode.capitalize()}")  # update GUI text
        print(f"[GUI] Switched control mode to: {mode}")

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
      - sends the gesture name to Profile / Actions
    """

    def __init__(self, data_dir, active_profile_name="1", control_mode="right"):
        self.data_dir = data_dir
        self.control_mode = control_mode.lower()   # "auto", "right", "left"

        # Model
        self.model = GestureKNNModel(data_dir=self.data_dir, k_neighbors=K_NEIGHBORS)

        # GUI
        self.gui = ClickTesterGUI(self)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

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

    # ---------- MOUSE MOVEMENT ----------
    PUL = ctypes.POINTER(ctypes.c_ulong)
    class Input_I(ctypes.Union):
        _fields_ = [("mi", MouseInput)]
    class Input(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("ii", Input_I)
        ]
    # --- Raw relative mouse movement (fastest possible) ---
    def move_mouse_raw(dx, dy):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

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

    # ---------- HAND MODE ----------

    def _select_pointer_and_action(self, hands_info):
        """
        Decide pointer and action hands based on control_mode:

        control_mode = "right":
            - Pointer: ONLY a 'Right' hand (highest confidence).
            - Action: ONLY a 'Left' hand (highest confidence),
                      gesture != 'point', above threshold.

        control_mode = "left":
            - Pointer: ONLY a 'Left' hand (highest confidence).
            - Action: ONLY a 'Right' hand (highest confidence),
                      gesture != 'point', above threshold.

        control_mode = "auto":
            - Pointer: any hand with 'point' gesture & confidence >= threshold,
                       otherwise best hand by confidence.
            - Action: must be a *different* hand from pointer, gesture != 'point',
                      above threshold.
        """
        pointer_hand = None
        action_hand = None
        action_label = "none"
        action_conf = 0.0

        if not hands_info:
            return pointer_hand, action_hand, action_label, action_conf

        mode = self.control_mode.lower()

        # ---------- RIGHT-HANDED MODE ----------
        if mode == "right":
            right_hands = [hi for hi in hands_info if hi["mp_label"] == "Right"]
            left_hands  = [hi for hi in hands_info if hi["mp_label"] == "Left"]

            # Pointer: ONLY right hand. If none, NO pointer.
            if right_hands:
                pointer_hand = max(right_hands, key=lambda hi: hi["raw_conf"])
            else:
                pointer_hand = None  # <- do NOT fall back to left hand

            # Action: ONLY from left hands, never the pointer, not 'point', above threshold
            for hi in left_hands:
                if hi is pointer_hand:
                    continue
                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                    continue
                if hi["raw_label"] == "point":
                    continue
                if hi["raw_conf"] > action_conf:
                    action_conf = hi["raw_conf"]
                    action_label = hi["raw_label"]
                    action_hand = hi

            return pointer_hand, action_hand, action_label, action_conf

        # ---------- LEFT-HANDED MODE ----------
        if mode == "left":
            right_hands = [hi for hi in hands_info if hi["mp_label"] == "Right"]
            left_hands  = [hi for hi in hands_info if hi["mp_label"] == "Left"]

            # Pointer: ONLY left hand. If none, NO pointer.
            if left_hands:
                pointer_hand = max(left_hands, key=lambda hi: hi["raw_conf"])
            else:
                pointer_hand = None  # <- do NOT fall back to right hand

            # Action: ONLY from right hands, never the pointer, not 'point', above threshold
            for hi in right_hands:
                if hi is pointer_hand:
                    continue
                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                    continue
                if hi["raw_label"] == "point":
                    continue
                if hi["raw_conf"] > action_conf:
                    action_conf = hi["raw_conf"]
                    action_label = hi["raw_label"]
                    action_hand = hi

            return pointer_hand, action_hand, action_label, action_conf

        # ---------- AUTO MODE ----------
        # Pointer: prefer any confident "point" gesture
        point_candidates = [
            hi for hi in hands_info
            if hi["raw_label"] == "point" and hi["raw_conf"] >= GESTURE_CONF_THRESHOLD
        ]

        if point_candidates:
            pointer_hand = max(point_candidates, key=lambda hi: hi["raw_conf"])
        else:
            # Fallback: highest-confidence hand
            pointer_hand = max(hands_info, key=lambda hi: hi["raw_conf"])

        # Action: must be a *different* hand from pointer, not 'point', above threshold
        if len(hands_info) >= 2:
            for hi in hands_info:
                if hi is pointer_hand:
                    continue  # enforce: one hand cannot be both pointer and action
                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                    continue
                if hi["raw_label"] == "point":
                    continue
                if hi["raw_conf"] > action_conf:
                    action_conf = hi["raw_conf"]
                    action_label = hi["raw_label"]
                    action_hand = hi

        return pointer_hand, action_hand, action_label, action_conf

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
        # Track previous gesture & action for transition logic
        prev_gesture = "none"
        prev_hold_action = None  # Actions object currently holding, if any

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

                    # ----- Pointer & Action selection based on mode ("auto", "right", "left") -----
                    pointer_hand, action_hand, action_label, action_conf = \
                        self._select_pointer_and_action(hands_info)

                    # ----- Move mouse with pointer hand -----
                    pointer_debug = "None"
                    if pointer_hand is not None:
                        fx, fy = pointer_hand["tip_px"]
                        nx, ny = pointer_hand["tip_norm"]

                        # draw fingertip ONLY for pointer hand
                        cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

                        sx = int(nx * self.screen_w)
                        sy = int(ny * self.screen_h)
                        move_mouse_absolute(sx, sy)

                        pointer_debug = (
                            f"{pointer_hand['mp_label']} "
                            f"({pointer_hand['raw_label']}, {pointer_hand['raw_conf']:.2f})"
                        )

                    # ----- Draw action hand (if any) -----
                    action_debug = "None"
                    if action_hand is not None:
                        action_debug = (
                            f"{action_hand['mp_label']} "
                            f"({action_label}, {action_conf:.2f})"
                        )
                        # ONLY draw skeleton for the action hand, NO red dot
                        self.mp_drawing.draw_landmarks(
                            frame,
                            action_hand["hand_lms"],
                            self.mp_hands.HAND_CONNECTIONS,
                        )
                    else:
                        action_label = "none"
                        action_conf = 0.0

                    # ===== CURRENT GESTURE (no smoothing) =====
                    current_gesture = action_label

                    # ===== Map gesture -> Actions object =====
                    mapped_action_obj = None

                    if current_gesture != "none":
                        # 1) look through all actions in the active profile
                        try:
                            actions_list = self.active_profile.getActionList()
                        except AttributeError:
                            actions_list = []

                        for a in actions_list:
                            try:
                                name = a.getName()
                                key = a.getKeyPressed()
                            except AttributeError:
                                continue

                            if name == current_gesture or key == current_gesture:
                                mapped_action_obj = a
                                break

                        # 2) fallback: getAction by name
                        if mapped_action_obj is None:
                            try:
                                direct_action = self.active_profile.getAction(current_gesture)
                                if direct_action is not None:
                                    mapped_action_obj = direct_action
                            except AttributeError:
                                pass

                    # ===== Handle transition logic: Click vs Hold =====
                    # If we had a previous hold action and gesture changed away, stop it
                    if prev_hold_action is not None:
                        if current_gesture != prev_hold_action.getName():
                            prev_hold_action.stopHold()
                            prev_hold_action = None

                    # Now handle the *current* gesture
                    if current_gesture != "none" and mapped_action_obj is not None:
                        input_type = mapped_action_obj.getInputType()

                        # CLICK -> only on transition (prev != current)
                        if input_type == "Click":
                            if current_gesture != prev_gesture:
                                print(f"[ACTION] Single click for '{current_gesture}'")
                                mapped_action_obj.useAction(mapped_action_obj.getName())

                        # HOLD -> call every frame; Actions takes care of starting thread once
                        elif input_type == "Hold":
                            mapped_action_obj.useAction(mapped_action_obj.getName())
                            prev_hold_action = mapped_action_obj

                        # Anything else (e.g. default with None) -> do nothing
                    else:
                        # No valid mapped action; any previous hold is already handled above
                        pass

                    # Update GUI with last gesture
                    self.gui.set_last_gesture(current_gesture)

                    # ----- Debug overlay -----
                    debug_text = (
                        f"Pointer: {pointer_debug} | "
                        f"Action: {action_debug} | "
                        f"Gesture: {current_gesture}"
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

                    # update previous for next loop
                    prev_gesture = current_gesture

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
    # Change control_mode to "right", "left", or "auto"
    app = GestureControllerApp(DATA_DIR, active_profile_name="1", control_mode="right")
    app.run()
