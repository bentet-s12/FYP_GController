import os
import time
import json
import numpy as np
import cv2
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import ctypes
from ctypes import wintypes
import pyautogui  # Cursor mode absolute positioning

from Actions import Actions  # your pydirectinput-based Actions.py

# ================== PATH CONFIG ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")

# IMPORTANT: this must match your real file name
PROFILE_JSON_PATH = os.path.join(SCRIPT_DIR, "profile_1.json")

# ================== CONSTANTS ====================

K_NEIGHBORS = 3
GESTURE_CONF_THRESHOLD = 0.6

# ================== RAW MOUSE (SendInput relative move) ==================

if hasattr(wintypes, "ULONG_PTR"):
    ULONG_PTR = wintypes.ULONG_PTR
else:
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class RawMouse:
    @staticmethod
    def move_rel(dx: int, dy: int):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = MOUSEINPUT(
            dx=int(dx),
            dy=int(dy),
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE,
            time=0,
            dwExtraInfo=0,
        )
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    @staticmethod
    def set_pos(x: int, y: int):
        ctypes.windll.user32.SetCursorPos(int(x), int(y))


# =================================================
#   HAND FEATURE EXTRACTOR
# =================================================

class HandFeatureExtractor:
    @staticmethod
    def landmarks_to_vec_pair(hand_lms):
        xs, ys = [], []
        for lm in hand_lms.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        xs -= xs[0]
        ys -= ys[0]

        radii = np.sqrt(xs**2 + ys**2)
        max_r = float(np.max(radii))
        if max_r > 0:
            xs_norm = xs / max_r
            ys_norm = ys / max_r
        else:
            xs_norm = xs
            ys_norm = ys

        vec_orig = np.concatenate([xs_norm, ys_norm], axis=0)
        vec_mirror = np.concatenate([-xs_norm, ys_norm], axis=0)
        return vec_orig, vec_mirror


# =================================================
#   KNN MODEL
# =================================================

class GestureKNNModel:
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
            raise FileNotFoundError(f"Could not find X.npy / y.npy / class_names.npy in {self.data_dir}")

        X = np.load(X_path)
        y = np.load(y_path)
        self.class_names = np.load(classes_path, allow_pickle=True)

        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn.fit(X, y)
        self.id_to_name = {i: name for i, name in enumerate(self.class_names)}

        print("[MODEL] Loaded:", X.shape, "| classes:", list(self.class_names))

    def classify_landmarks(self, hand_lms):
        vec_orig, vec_mirror = HandFeatureExtractor.landmarks_to_vec_pair(hand_lms)

        best_label = "none"
        best_conf = -1.0

        for vec in (vec_orig, vec_mirror):
            proba = self.knn.predict_proba([vec])[0]
            top_idx = int(np.argmax(proba))
            top_conf = float(proba[top_idx])
            if top_conf > best_conf:
                best_conf = top_conf
                best_label = self.id_to_name[top_idx]

        return best_label, best_conf


# =================================================
#   PROFILE JSON LOADER (BYPASS ProfileManager)
# =================================================

def load_actions_from_profile_json(profile_path: str):
    """
    Loads profile_1.json (your format) and returns:
      action_map: dict[name -> Actions]
    """
    action_map = {}

    if not os.path.exists(profile_path):
        print(f"[PROFILE] Missing: {profile_path}")
        return action_map

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        actions = data.get("Actions", [])
        for a in actions:
            name = a.get("name")
            key = a.get("key_pressed")
            input_type = a.get("input_type")

            # skip empty/default placeholders
            if not name or name == "default":
                continue
            if not key or not input_type:
                continue

            action_map[name] = Actions(name, key, input_type)

        print(f"[PROFILE] Loaded {len(action_map)} actions from {os.path.basename(profile_path)}: {list(action_map.keys())}")
        return action_map

    except Exception as e:
        print("[PROFILE] Failed to load:", e)
        return action_map


# =================================================
#   GUI
# =================================================

class ClickTesterGUI:
    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Gesture Controller Tester")
        self.root.geometry("460x420")

        self.click_count = 0
        self.click_var = tk.StringVar(value="Clicks: 0")
        self.gesture_var = tk.StringVar(value="Last gesture: none")
        self.mode_var = tk.StringVar(value=self._mode_text())

        # Live camera sensitivity (gain)
        self.cam_sens_var = tk.StringVar(value=f"{self.app._joy_gain:.1f}")

        self._build_widgets()

    def _mode_text(self):
        return (
            f"Hand Mode: {self.app.control_mode.capitalize()} | "
            f"Mouse Mode: {self.app.mouse_mode} | "
            f"Cam Sens: {self.app._joy_gain:.1f}"
        )

    def _apply_camera_sensitivity(self):
        raw = self.cam_sens_var.get().strip()
        try:
            v = float(raw)
        except ValueError:
            self.cam_sens_var.set(f"{self.app._joy_gain:.1f}")
            return

        # clamp to sane range
        if v < 1:
            v = 1.0
        if v > 9999:
            v = 9999.0

        self.app._joy_gain = float(v)
        self.cam_sens_var.set(f"{self.app._joy_gain:.1f}")
        self.mode_var.set(self._mode_text())

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

        tk.Label(self.root, textvariable=self.gesture_var, font=("Arial", 12)).pack(pady=3)

        tk.Label(
            self.root,
            textvariable=self.mode_var,
            font=("Arial", 12, "bold"),
            fg="blue",
        ).pack(pady=3)

        # Row 1: hand mode buttons
        hand_frame = tk.Frame(self.root)
        hand_frame.pack(fill="x", pady=6)

        tk.Button(hand_frame, text="Left-handed", command=lambda: self.set_hand_mode("left")).pack(
            side="left", expand=True, padx=6
        )
        tk.Button(hand_frame, text="Right-handed", command=lambda: self.set_hand_mode("right")).pack(
            side="left", expand=True, padx=6
        )
        tk.Button(hand_frame, text="Auto", command=lambda: self.set_hand_mode("auto")).pack(
            side="left", expand=True, padx=6
        )

        # Row 2: mouse mode buttons
        mouse_frame = tk.Frame(self.root)
        mouse_frame.pack(fill="x", pady=6)

        tk.Button(mouse_frame, text="Camera Mode", command=lambda: self.set_mouse_mode("CAMERA")).pack(
            side="left", expand=True, padx=6
        )
        tk.Button(mouse_frame, text="Cursor Mode", command=lambda: self.set_mouse_mode("CURSOR")).pack(
            side="left", expand=True, padx=6
        )

        # Row 3: Camera sensitivity input (LIVE)
        sens_frame = tk.Frame(self.root)
        sens_frame.pack(fill="x", pady=10)

        tk.Label(sens_frame, text="Camera Sensitivity:", font=("Arial", 11)).pack(side="left", padx=8)

        spin = tk.Spinbox(
            sens_frame,
            from_=1,
            to=9999,
            increment=1,
            width=8,
            textvariable=self.cam_sens_var,
            command=self._apply_camera_sensitivity,  # arrow clicks
            font=("Arial", 11),
        )
        spin.pack(side="left", padx=6)

        tk.Button(sens_frame, text="Apply", command=self._apply_camera_sensitivity).pack(side="left", padx=6)

        spin.bind("<Return>", lambda e: self._apply_camera_sensitivity())
        spin.bind("<FocusOut>", lambda e: self._apply_camera_sensitivity())

        tk.Label(self.root, text="Tip: Higher = faster in CAMERA mode", font=("Arial", 10), fg="gray").pack(pady=2)

        # optional reload button (very useful)
        tk.Button(self.root, text="Reload profile_1.json", command=self.app.reload_profile_actions).pack(pady=6)

    def set_hand_mode(self, mode: str):
        self.app.control_mode = mode.lower()
        self.app._reset_mouse_state()
        self.mode_var.set(self._mode_text())

    def set_mouse_mode(self, mode: str):
        self.app.mouse_mode = mode
        self.app._reset_mouse_state()
        self.mode_var.set(self._mode_text())

    def increment_click_counter(self):
        self.click_count += 1
        self.click_var.set(f"Clicks: {self.click_count}")

    def set_last_gesture(self, gesture_name: str):
        self.gesture_var.set(f"Last gesture: {gesture_name}")

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# =================================================
#   MAIN APP
# =================================================

class GestureControllerApp:
    """
    control_mode: "right" | "left" | "auto"
    mouse_mode: "CAMERA" | "CURSOR"
    """

    def __init__(self, data_dir, control_mode="right", mouse_mode="CAMERA"):
        self.data_dir = data_dir
        self.control_mode = control_mode.lower()
        self.mouse_mode = mouse_mode  # "CAMERA" or "CURSOR"

        self.model = GestureKNNModel(data_dir=self.data_dir, k_neighbors=K_NEIGHBORS)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Cannot open webcam.")

        self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)

        # CAMERA mode (relative joystick-style) --- set defaults BEFORE GUI
        self._joy_dx = 0.0
        self._joy_dy = 0.0
        self._joy_alpha = 0.18
        self._joy_deadzone = 0.04
        self._joy_gain = 70.0          # default sensitivity (current)
        self._joy_max_step = 70
        # EXTRA smoothing for CAMERA mode (prevents jitter at high gain)
        self._step_ema_x = 0.0
        self._step_ema_y = 0.0
        self._step_alpha = 0.12   # smaller = smoother (0.06–0.20 is good)
        self._min_step_px = 2     # ignore tiny movement after smoothing

        # CURSOR mode (absolute moveTo + EMA)
        self._ema_nx = None
        self._ema_ny = None
        self._ema_alpha = 0.25
        self._cursor_deadzone_px = 2

        # LOAD ACTIONS FROM profile_1.json
        self.action_map = load_actions_from_profile_json(PROFILE_JSON_PATH)

        self.gui = ClickTesterGUI(self)

        self.prev_gesture = "none"
        self.prev_hold_action = None

        RawMouse.set_pos(self.screen_w // 2, self.screen_h // 2)

    def reload_profile_actions(self):
        self.action_map = load_actions_from_profile_json(PROFILE_JSON_PATH)

    # ---------- mouse helpers ----------

    def _reset_mouse_state(self):
        self._joy_dx = 0.0
        self._joy_dy = 0.0
        self._ema_nx = None
        self._ema_ny = None
        # reset camera step smoothing
        self._step_ema_x = 0.0
        self._step_ema_y = 0.0

    def _mouse_move_camera(self, nx: float, ny: float):
        dx = nx - 0.5
        dy = ny - 0.5

        # deadzone in normalized space
        if abs(dx) < self._joy_deadzone:
            dx = 0.0
        if abs(dy) < self._joy_deadzone:
            dy = 0.0

        # 1) Smooth normalized joystick direction
        a = self._joy_alpha
        self._joy_dx = (1 - a) * self._joy_dx + a * dx
        self._joy_dy = (1 - a) * self._joy_dy + a * dy

        # raw step from sensitivity
        raw_step_x = self._joy_dx * self._joy_gain
        raw_step_y = self._joy_dy * self._joy_gain

        # clamp BEFORE step smoothing (prevents wild swings)
        raw_step_x = max(-self._joy_max_step, min(self._joy_max_step, raw_step_x))
        raw_step_y = max(-self._joy_max_step, min(self._joy_max_step, raw_step_y))

        # 2) Smooth pixel steps (this is what kills jitter at high sensitivity)
        b = self._step_alpha
        self._step_ema_x = (1 - b) * self._step_ema_x + b * raw_step_x
        self._step_ema_y = (1 - b) * self._step_ema_y + b * raw_step_y

        step_x = int(round(self._step_ema_x))
        step_y = int(round(self._step_ema_y))

        # 3) Deadband in pixel space (ignore tiny oscillations)
        if abs(step_x) < self._min_step_px:
            step_x = 0
        if abs(step_y) < self._min_step_px:
            step_y = 0

        if step_x == 0 and step_y == 0:
            return

        RawMouse.move_rel(step_x, step_y)


    def _mouse_move_cursor(self, nx: float, ny: float):
        if self._ema_nx is None:
            self._ema_nx, self._ema_ny = nx, ny
        else:
            a = self._ema_alpha
            self._ema_nx = (1 - a) * self._ema_nx + a * nx
            self._ema_ny = (1 - a) * self._ema_ny + a * ny

        tx = int(self._ema_nx * self.screen_w)
        ty = int(self._ema_ny * self.screen_h)

        cx, cy = pyautogui.position()
        if abs(tx - cx) <= self._cursor_deadzone_px and abs(ty - cy) <= self._cursor_deadzone_px:
            return

        pyautogui.moveTo(tx, ty, duration=0)

    def _move_mouse(self, nx: float, ny: float):
        if self.mouse_mode == "CURSOR":
            self._mouse_move_cursor(nx, ny)
        else:
            self._mouse_move_camera(nx, ny)

    # ---------- hand info ----------

    def _get_hands_info(self, results, frame_w, frame_h):
        infos = []
        if not (results.multi_hand_landmarks and results.multi_handedness):
            return infos

        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_label = handedness.classification[0].label
            tip = hand_lms.landmark[8]
            fx = int(tip.x * frame_w)
            fy = int(tip.y * frame_h)

            infos.append(
                {
                    "hand_lms": hand_lms,
                    "mp_label": mp_label,
                    "tip_px": (fx, fy),
                    "tip_norm": (float(tip.x), float(tip.y)),
                    "raw_label": "none",
                    "raw_conf": 0.0,
                }
            )
        return infos

    # ---------- STRICT mode selection ----------

    def _select_pointer_and_action(self, hands_info):
        pointer_hand = None
        action_hand = None
        action_label = "none"
        action_conf = 0.0

        if not hands_info:
            return pointer_hand, action_hand, action_label, action_conf

        mode = self.control_mode

        if mode == "right":
            right_hands = [hi for hi in hands_info if hi["mp_label"] == "Right"]
            left_hands = [hi for hi in hands_info if hi["mp_label"] == "Left"]

            pointer_hand = max(right_hands, key=lambda hi: hi["raw_conf"]) if right_hands else None

            for hi in left_hands:
                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                    continue
                if hi["raw_label"] == "point":
                    continue
                if hi["raw_conf"] > action_conf:
                    action_conf = hi["raw_conf"]
                    action_label = hi["raw_label"]
                    action_hand = hi

            return pointer_hand, action_hand, action_label, action_conf

        if mode == "left":
            left_hands = [hi for hi in hands_info if hi["mp_label"] == "Left"]
            right_hands = [hi for hi in hands_info if hi["mp_label"] == "Right"]

            pointer_hand = max(left_hands, key=lambda hi: hi["raw_conf"]) if left_hands else None

            for hi in right_hands:
                if hi["raw_conf"] < GESTURE_CONF_THRESHOLD:
                    continue
                if hi["raw_label"] == "point":
                    continue
                if hi["raw_conf"] > action_conf:
                    action_conf = hi["raw_conf"]
                    action_label = hi["raw_label"]
                    action_hand = hi

            return pointer_hand, action_hand, action_label, action_conf

        # AUTO
        point_candidates = [
            hi for hi in hands_info
            if hi["raw_label"] == "point" and hi["raw_conf"] >= GESTURE_CONF_THRESHOLD
        ]
        pointer_hand = max(point_candidates, key=lambda hi: hi["raw_conf"]) if point_candidates else max(
            hands_info, key=lambda hi: hi["raw_conf"]
        )

        if len(hands_info) >= 2:
            for hi in hands_info:
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

    # ---------- Run loop ----------

    def run(self):
        try:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            ) as hands:
                while True:
                    self.gui.update()

                    ret, frame = self.cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    hands_info = self._get_hands_info(results, w, h)

                    for info in hands_info:
                        raw_label, raw_conf = self.model.classify_landmarks(info["hand_lms"])
                        info["raw_label"] = raw_label
                        info["raw_conf"] = raw_conf

                    pointer_hand, action_hand, action_label, action_conf = self._select_pointer_and_action(hands_info)

                    # pointer -> mouse
                    pointer_debug = "None"
                    if pointer_hand is not None:
                        fx, fy = pointer_hand["tip_px"]
                        nx, ny = pointer_hand["tip_norm"]
                        cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
                        self._move_mouse(nx, ny)
                        pointer_debug = f"{pointer_hand['mp_label']} ({pointer_hand['raw_label']}, {pointer_hand['raw_conf']:.2f})"

                    # action drawing
                    action_debug = "None"
                    if action_hand is not None:
                        action_debug = f"{action_hand['mp_label']} ({action_label}, {action_conf:.2f})"
                        self.mp_drawing.draw_landmarks(frame, action_hand["hand_lms"], self.mp_hands.HAND_CONNECTIONS)
                    else:
                        action_label = "none"

                    current_gesture = action_label
                    self.gui.set_last_gesture(current_gesture)

                    # === MAP gesture -> action (DIRECT JSON MAP) ===
                    mapped_action_obj = self.action_map.get(current_gesture)

                    # ---- FORCE STOP HOLD when gesture is none OR changed OR unmapped ----
                    if self.prev_hold_action is not None:
                        if current_gesture == "none" or mapped_action_obj is None or current_gesture != self.prev_hold_action.getName():
                            self.prev_hold_action.stopHold()
                            self.prev_hold_action = None

                    fired_text = "NO"
                    mapped_text = "None"

                    if current_gesture != "none" and mapped_action_obj is not None:
                        input_type = mapped_action_obj.getInputType()
                        key = mapped_action_obj.getKeyPressed()
                        mapped_text = f"{current_gesture} -> {key} ({input_type})"

                        if input_type == "Click":
                            if current_gesture != self.prev_gesture:
                                mapped_action_obj.useAction(mapped_action_obj.getName())
                                fired_text = "CLICK"
                                print(f"[FIRE] CLICK {current_gesture} -> {key}")
                        elif input_type == "Hold":
                            mapped_action_obj.useAction(mapped_action_obj.getName())
                            self.prev_hold_action = mapped_action_obj
                            fired_text = "HOLD"
                    else:
                        pass

                    debug_text = f"Pointer: {pointer_debug} | Action: {action_debug} | Mapped: {mapped_text} | Fired: {fired_text}"
                    cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                    cv2.imshow("Gesture Pointer Prototype (Camera/Cursor Toggle)", frame)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    elif k == ord("r"):
                        self.reload_profile_actions()

                    self.prev_gesture = current_gesture

        finally:
            self._cleanup()

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

        if self.prev_hold_action is not None:
            try:
                self.prev_hold_action.stopHold()
            except Exception:
                pass

        self.gui.destroy()


# =================================================
#   MAIN
# =================================================

if __name__ == "__main__":
    app = GestureControllerApp(DATA_DIR, control_mode="right", mouse_mode="CURSOR")
    app.run()
