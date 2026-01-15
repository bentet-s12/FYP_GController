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
import threading
from screeninfo import get_monitors



from Actions import Actions  # your pydirectinput-based Actions.py
# ================== PATH CONFIG ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")

# IMPORTANT: this must match your real file name
PROFILE_JSON_PATH = os.path.join(SCRIPT_DIR, "profile_1.json")
GESTURELIST_JSON_PATH = os.path.join(SCRIPT_DIR, "GestureList.json")
STRICT_GESTURELIST = True  # if True: ignore profile mappings whose gesture is not in GestureList.json

# ================== CONSTANTS ====================

K_NEIGHBORS = 3
GESTURE_CONF_THRESHOLD = 0.6

# ================== CURSOR / CAMERA MOUSE ==================

# Win32 SendInput for relative mouse movement (CAMERA mode)
user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("i",)
    _fields_ = [("type", wintypes.DWORD),
                ("i", _I)]

def send_relative_mouse(dx, dy):
    inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=int(dx), dy=int(dy), mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# =================================================
#   KNN CLASSIFIER
# =================================================

class KNNGestureClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        self.class_names = []
        self.trained = False
        self.id_to_name = {}

    def load_dataset(self):
        X_path = os.path.join(DATA_DIR, "X.npy")
        y_path = os.path.join(DATA_DIR, "y.npy")
        class_path = os.path.join(DATA_DIR, "class_names.npy")

        if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(class_path)):
            print("[KNN] Dataset files missing.")
            return False

        X = np.load(X_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        self.class_names = np.load(class_path, allow_pickle=True).tolist()

        # ---------- normalize shapes ----------
        # X should be (N, D)
        if X is None or len(X) == 0:
            print("[KNN] X is empty.")
            return False

        # y can be (N,), (N,1), (N,C), or weird object arrays
        y = np.array(y, dtype=object)

        # If y is 2D one-hot/proba: (N, C) -> class indices
        if y.ndim == 2 and y.shape[1] > 1:
            try:
                y_num = np.asarray(y, dtype=float)
                y_idx = np.argmax(y_num, axis=1).astype(int)
                y = y_idx
                print("[KNN] Detected one-hot/probability y. Converted using argmax.")
            except Exception:
                pass

        # If y is (N,1) -> (N,)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        # ---------- convert label types ----------
        # If y is numeric floats, cast to int if safe
        if y.dtype != object:
            # numeric arrays end up here
            if y.dtype.kind in ("f", "i", "u"):
                # if float but basically ints, cast
                if y.dtype.kind == "f":
                    y_float = y.astype(float)
                    if np.all(np.isfinite(y_float)) and np.all(np.abs(y_float - np.round(y_float)) < 1e-6):
                        y = np.round(y_float).astype(int)
                        print("[KNN] y looked like float-integers. Cast to int.")
                else:
                    y = y.astype(int)

        # If y is object array, it may contain numpy scalars or lists
        # Try to unwrap scalars
        if y.dtype == object:
            y2 = []
            for v in y:
                # unwrap 0-d arrays / numpy scalars
                if isinstance(v, np.ndarray) and v.shape == ():
                    v = v.item()
                # unwrap single-element lists/tuples/arrays
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                    v = v[0]
                y2.append(v)
            y = np.array(y2, dtype=object)

            # If after unwrap it’s numeric, try convert
            try:
                y_float = np.array(y, dtype=float)
                if np.all(np.isfinite(y_float)) and np.all(np.abs(y_float - np.round(y_float)) < 1e-6):
                    y = np.round(y_float).astype(int)
                    print("[KNN] y object contained numeric float-integers. Cast to int.")
            except Exception:
                pass

        # ---------- map numeric labels -> class names ----------
        if isinstance(y, np.ndarray) and y.dtype.kind in ("i", "u"):
            # y is integer class indices
            # Prefer class_names.npy mapping if it matches
            if isinstance(self.class_names, list) and len(self.class_names) > 0:
                max_idx = int(np.max(y))
                if max_idx < len(self.class_names):
                    y = np.array([self.class_names[int(i)] for i in y], dtype=object)
                    print("[KNN] Mapped numeric y -> class_names strings.")
                else:
                    print("[KNN] WARNING: y has indices beyond class_names length. Keeping numeric labels.")
            else:
                print("[KNN] WARNING: class_names is empty; keeping numeric labels.")

        # ---------- final sanity ----------
        unique_preview = list(dict.fromkeys(map(str, y.tolist())))[:20] if hasattr(y, "tolist") else []
        print(f"[KNN] X shape={X.shape}, y shape={np.shape(y)}, y sample(unique<=20)={unique_preview}")

        # ---- FINAL NORMALIZATION: force ALL labels to strings ----
        y = np.array([str(v) for v in y], dtype=object)

        # Fit KNN
        self.knn.fit(X, y)
        self.trained = True

        # build label set
        labels = sorted(set(list(y)))
        self.id_to_name = {i: labels[i] for i in range(len(labels))}

        print(f"[KNN] Loaded dataset: {len(X)} samples, {len(labels)} labels")
        return True


    def predict(self, feat_vec):
        if not self.trained:
            return "none", 0.0

        # KNN predict proba
        proba = self.knn.predict_proba([feat_vec])[0]
        top_idx = int(np.argmax(proba))
        top_conf = float(proba[top_idx])

        # KNN uses y labels directly, easiest get predicted label:
        pred_label = self.knn.predict([feat_vec])[0]

        return pred_label, top_conf


# =================================================
#   FEATURE VECTOR (42D)
# =================================================

def landmarks_to_feature_vector(hand_lm, mirror=False):
    coords = []
    for p in hand_lm.landmark:
        coords.append([p.x, p.y])
    coords = np.array(coords, dtype=np.float32)

    # wrist-centered
    wrist = coords[0].copy()
    coords = coords - wrist

    # scale normalized
    scale = np.linalg.norm(coords[9]) + 1e-6
    coords = coords / scale

    # optional mirror for left/right symmetry
    if mirror:
        coords[:, 0] *= -1.0

    return coords.reshape(-1)


# =================================================
#   PROFILE JSON LOADER (BYPASS ProfileManager)
# =================================================

# ===================== OLD PROFILE LOADER (COMMENTED OUT) =====================
# def load_actions_from_profile_json(profile_path: str):
#     """
#     Loads profile_1.json (your format) and returns:
#       action_map: dict[name -> Actions]
#     """
#     action_map = {}
#
#     if not os.path.exists(profile_path):
#         print(f"[PROFILE] Missing: {profile_path}")
#         return action_map
#
#     try:
#         with open(profile_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         actions = data.get("Actions", [])
#         for a in actions:
#             name = a.get("name")
#             key = a.get("key_pressed")
#             input_type = a.get("input_type")
#             key_type = a.get("key_type")
#
#             # skip empty/default placeholders
#             if not name or name == "default":
#                 continue
#             if not key or not input_type:
#                 continue
#
#             action_map[name] = Actions(name, key, input_type, key_type)
#
#         print(f"[PROFILE] Loaded {len(action_map)} actions from {os.path.basename(profile_path)}: {list(action_map.keys())}")
#         return action_map
#
#     except Exception as e:
#         print("[PROFILE] Failed to load:", e)
#         return action_map

# ===================== NEW PROFILE LOADER (ACTIVE) =====================

def load_gesture_list(gesturelist_path: str) -> list[str]:
    """
    Loads GestureList.json (a JSON list of gesture names).
    Returns a normalized unique list (order preserved).
    """
    if not os.path.exists(gesturelist_path):
        print(f"[GestureList] Missing: {gesturelist_path} (treating as empty)")
        return []
    try:
        with open(gesturelist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("[GestureList] Invalid format (must be a JSON list). Treating as empty.")
            return []
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
    except Exception as e:
        print("[GestureList] Failed to load:", e)
        return []


def load_actions_from_profile_json(profile_path: str):
    """
    NEW LOGIC:
      - GestureList.json is the source of truth for valid gestures.
      - profile_<id>.json contains mappings keyed by 'gesture' (preferred), with legacy fallback to 'name'.
      - action_map is built as dict[gesture -> Actions].

    Expected mapping entry (new schema):
      {"gesture": "thumbs_up", "key_pressed": "space", "input_type": "Click", "key_type": "Keyboard"}

    Legacy fallback:
      - If 'gesture' is missing, we treat 'name' as the gesture label.
    """
    action_map = {}

    gesture_list = load_gesture_list(GESTURELIST_JSON_PATH)
    gesture_set = set(gesture_list)

    if not os.path.exists(profile_path):
        print(f"[PROFILE] Missing: {profile_path}")
        return action_map

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        actions = data.get("Actions", [])
        for a in actions:
            if not isinstance(a, dict):
                continue

            gesture = a.get("gesture") or a.get("name")  # legacy fallback
            if not isinstance(gesture, str):
                continue
            gesture = gesture.strip()

            # skip empty/default placeholders
            if not gesture or gesture == "default":
                continue

            if STRICT_GESTURELIST and gesture_set and (gesture not in gesture_set):
                print(f"[PROFILE] Skip mapping: gesture '{gesture}' not in GestureList.json")
                continue

            key = a.get("key_pressed")
            input_type = a.get("input_type")
            key_type = a.get("key_type")

            if not key or not input_type:
                continue

            # normalize
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

            # IMPORTANT: Actions.__init__(name, G_name, key_pressed, input_type, key_type)
            action_obj = Actions(
                name=gesture,
                G_name=gesture,
                key_pressed=key,
                input_type=input_type,
                key_type=key_type
            )

            action_map[gesture] = action_obj

        print(f"[PROFILE] Loaded {len(action_map)} mappings from {os.path.basename(profile_path)}: {list(action_map.keys())}")
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
        self.root.geometry("780x620")
        self.root.resizable(True, True)

        self.main = tk.Frame(self.root)
        self.main.pack(fill="both", expand=True, padx=10, pady=10)

        self.click_count = 0
        self.label_mode = tk.Label(self.main, text="Hand Mode: right | Mouse Mode: CAMERA", font=("Arial", 12))
        self.label_mode.pack(pady=8)

        self.label_gesture = tk.Label(self.main, text="Gesture: none", font=("Arial", 14))
        self.label_gesture.pack(pady=8)

        self.label_mapped = tk.Label(self.main, text="Mapped: None", font=("Arial", 12))
        self.label_mapped.pack(pady=8)

        self.label_fired = tk.Label(self.main, text="Fired: NO", font=("Arial", 12))
        self.label_fired.pack(pady=8)

        self.btn_reload = tk.Button(self.main, text="Reload Profile (r)", command=self.app.reload_profile_actions)
        self.btn_reload.pack(pady=10)

        # Hand mode buttons
        frame_hand = tk.Frame(self.main)
        frame_hand.pack(fill="x", expand=True, pady=6)
        tk.Label(frame_hand, text="Hand Mode: ").pack(side=tk.LEFT)
        tk.Button(frame_hand, text="Right", command=lambda: self.app.set_hand_mode("right")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="Left", command=lambda: self.app.set_hand_mode("left")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="Auto", command=lambda: self.app.set_hand_mode("auto")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="MultiKB", command=lambda: self.app.set_hand_mode("multi_keyboard")).pack(side=tk.LEFT, padx=4)

        # Mouse mode buttons
        frame_mouse = tk.Frame(self.main)
        frame_mouse.pack(fill="x", expand=True, pady=6)
        tk.Label(frame_mouse, text="Mouse Mode: ").pack(side=tk.LEFT)
        tk.Button(frame_mouse, text="Disabled", command=lambda: self.app.set_mouse_mode("DISABLED")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mouse, text="Camera", command=lambda: self.app.set_mouse_mode("CAMERA")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mouse, text="Cursor", command=lambda: self.app.set_mouse_mode("CURSOR")).pack(side=tk.LEFT, padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- Camera Adjustments ----
        frame_cam = tk.LabelFrame(self.main, text="Camera")
        frame_cam.pack(fill="x", padx=10, pady=8)

        # Contrast slider
        tk.Label(frame_cam, text="Contrast").grid(row=0, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=self.app.cam_contrast)
        tk.Scale(
            frame_cam, from_=-0.5, to=3.0, resolution=0.05,
            orient="horizontal", variable=self.contrast_var,
            command=lambda _=None: self._on_contrast_change()
        ).grid(row=0, column=1, sticky="ew", padx=8)

        # Brightness slider (-100 to +100) (optional but useful)
        tk.Label(frame_cam, text="Brightness").grid(row=1, column=0, sticky="w")
        self.brightness_var = tk.IntVar(value=self.app.cam_brightness)
        tk.Scale(
            frame_cam, from_=-100, to=100, resolution=1,
            orient="horizontal", variable=self.brightness_var,
            command=lambda _=None: self._on_brightness_change()
        ).grid(row=1, column=1, sticky="ew", padx=8)

        # Grayscale toggle
        self.gray_var = tk.BooleanVar(value=self.app.cam_grayscale)
        tk.Checkbutton(
            frame_cam, text="Grayscale",
            variable=self.gray_var,
            command=self._on_gray_toggle
        ).grid(row=2, column=0, sticky="w", pady=(4,0))

        # Apply to tracking toggle (advanced)
        self.track_adj_var = tk.BooleanVar(value=self.app.cam_apply_to_tracking)
        tk.Checkbutton(
            frame_cam, text="Apply adjustments to tracking (advanced)",
            variable=self.track_adj_var,
            command=self._on_track_adj_toggle
        ).grid(row=2, column=1, sticky="w", pady=(4,0))

        frame_cam.columnconfigure(1, weight=1)

        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())



    def on_close(self):
        self.app.running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.destroy()


    def update_status(self, mode_text, gesture_text, mapped_text, fired_text):
        self.label_mode.config(text=mode_text)
        self.label_gesture.config(text=gesture_text)
        self.label_mapped.config(text=mapped_text)
        self.label_fired.config(text=fired_text)

    def _on_contrast_change(self):
        self.app.cam_contrast = float(self.contrast_var.get())

    def _on_brightness_change(self):
        self.app.cam_brightness = int(self.brightness_var.get())

    def _on_gray_toggle(self):
        self.app.cam_grayscale = bool(self.gray_var.get())

    def _on_track_adj_toggle(self):
        self.app.cam_apply_to_tracking = bool(self.track_adj_var.get())


# =================================================
#   MONITOR DETECTION
# =================================================
def select_monitor():
    monitors = get_monitors()

    if not monitors:
        print("[MONITOR] No monitors detected. Using full screen.")
        return None

    if len(monitors) == 1:
        m = monitors[0]
        print(f"[MONITOR] Single monitor detected: {m.width}x{m.height}")
        return m

    print("\nDetected monitors:")
    for i, m in enumerate(monitors):
        print(
            f" {i}) {m.width}x{m.height} "
            f"@ ({m.x}, {m.y})"
        )

    while True:
        sel = input("Select monitor index to use for CURSOR mode: ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(monitors):
                return monitors[idx]
        print("Invalid selection. Try again.")

# =================================================
#   MAIN APP
# =================================================

class GestureControllerApp:
    def __init__(self):
        self.running = True

        # modes
        self.hand_mode = "right"   # right/left/auto
        self.mouse_mode = "DISABLED" # DISABLED/CAMERA/CURSOR

        # Camera adjustment settings (GUI-controlled)
        self.cam_contrast = 1.0      # 0.5 .. 3.0
        self.cam_brightness = 0      # -100 .. +100
        self.cam_grayscale = False
        self.cam_apply_to_tracking = False  # recommended False

        # joystick (CAMERA mode)
        self._joy_gain = 70.0
        self._joy_alpha = 0.18
        self._joy_deadzone = 0.04
        self._joy_max_step = 70.0
        self._joy_sm_x = 0.0
        self._joy_sm_y = 0.0

        # KNN
        self.classifier = KNNGestureClassifier()
        self.classifier.load_dataset()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        self.cap = cv2.VideoCapture(0)

        # screen center for CURSOR mode
        self.monitor = select_monitor()

        if self.monitor:
            self.screen_x = self.monitor.x
            self.screen_y = self.monitor.y
            self.screen_w = self.monitor.width
            self.screen_h = self.monitor.height
        else:
            # fallback to full desktop
            screen_w, screen_h = pyautogui.size()
            self.screen_x = 0
            self.screen_y = 0
            self.screen_w = screen_w
            self.screen_h = screen_h

        self.screen_cx = self.screen_x + self.screen_w // 2
        self.screen_cy = self.screen_y + self.screen_h // 2


        # LOAD ACTIONS FROM profile_1.json (NEW loader)
        self.action_map = load_actions_from_profile_json(PROFILE_JSON_PATH)

        self.gui = ClickTesterGUI(self)

        # Per-hand state (for multi-keyboard mode and also safe for normal modes)
        self.prev_gesture_by_hand = {"Left": "none", "Right": "none"}
        self.prev_hold_action_by_hand = {"Left": None, "Right": None}

        # Optional: in multi_keyboard mode, only allow Keyboard actions (recommended)
        self.MULTI_KEYBOARD_ONLY = True

    def set_hand_mode(self, mode: str):
        self.hand_mode = mode

    def set_mouse_mode(self, mode: str):
        self.mouse_mode = mode
        if mode == "DISABLED":
            self._reset_mouse_state()

    def reload_profile_actions(self):
        self.action_map = load_actions_from_profile_json(PROFILE_JSON_PATH)

    # ---------- mouse helpers ----------

    def _reset_mouse_state(self):
        self._joy_sm_x = 0.0
        self._joy_sm_y = 0.0

    def _apply_deadzone(self, v):
        if abs(v) < self._joy_deadzone:
            return 0.0
        return v

    def _camera_move(self, dx, dy):
        # smooth
        self._joy_sm_x = (1 - self._joy_alpha) * self._joy_sm_x + self._joy_alpha * dx
        self._joy_sm_y = (1 - self._joy_alpha) * self._joy_sm_y + self._joy_alpha * dy

        sx = np.clip(self._joy_sm_x, -self._joy_max_step, self._joy_max_step)
        sy = np.clip(self._joy_sm_y, -self._joy_max_step, self._joy_max_step)

        send_relative_mouse(sx, sy)

    def _cursor_move(self, x_norm, y_norm):
        """
        Absolute cursor move constrained to selected monitor.
        x_norm, y_norm in [0..1]
        """
        x = int(self.screen_x + x_norm * self.screen_w)
        y = int(self.screen_y + y_norm * self.screen_h)

        # Clamp just in case
        x = max(self.screen_x, min(x, self.screen_x + self.screen_w - 1))
        y = max(self.screen_y, min(y, self.screen_y + self.screen_h - 1))

        pyautogui.moveTo(x, y)

     # ---------- keyboard helpers ----------

    def _process_action_for_hand(self, hand_label: str, current_gesture: str):
        """
        Execute mapped action for a given hand label ("Left"/"Right") based on current_gesture.
        Supports simultaneous holds on both hands.
        """
        prev_gesture = self.prev_gesture_by_hand.get(hand_label, "none")
        prev_hold = self.prev_hold_action_by_hand.get(hand_label)

        mapped_action_obj = self.action_map.get(current_gesture) if current_gesture != "none" else None

        # Optional restriction: MultiKB should only drive keyboard actions
        if self.hand_mode == "multi_keyboard" and self.MULTI_KEYBOARD_ONLY and mapped_action_obj is not None:
            if mapped_action_obj.getKeyType() != "Keyboard":
                mapped_action_obj = None

        # Stop hold if gesture changes/disappears/unmapped
        if prev_hold is not None:
            if current_gesture == "none" or mapped_action_obj is None or current_gesture != prev_hold.getName():
                print("[STOP]", hand_label, "cur=", current_gesture,
                "prev_hold_name=", prev_hold.getName(),
                "mapped=", None if mapped_action_obj is None else mapped_action_obj.getInputType(), flush=True)
                prev_hold.stopHold()
                prev_hold = None

        fired_text = "NO"
        mapped_text = "None"

        if mapped_action_obj is not None:
            input_type = mapped_action_obj.getInputType()
            key = mapped_action_obj.getKeyPressed()
            mapped_text = f"{current_gesture} -> {key} ({input_type})"

            # CLICK / DOUBLE CLICK fires once on transition
            if input_type in ("Click", "D_Click"):
                if current_gesture != prev_gesture:
                    mapped_action_obj.useAction(mapped_action_obj.getName())
                    fired_text = "YES" if input_type == "Click" else "D_CLICK"

            # HOLD repeats safely
            elif input_type == "Hold":
                if prev_hold is None:
                    print("###HOLD_START###", hand_label, current_gesture, key, flush=True)

                mapped_action_obj.useAction(mapped_action_obj.getName())
                prev_hold = mapped_action_obj
                fired_text = "HOLD"



        # Save state back
        self.prev_gesture_by_hand[hand_label] = current_gesture
        self.prev_hold_action_by_hand[hand_label] = prev_hold

        return mapped_text, fired_text
    
    def _apply_camera_adjustments(self, frame_bgr):
        """
        Apply brightness/contrast and optional grayscale for display/tracking.
        Returns BGR frame.
        """
        # Contrast/Brightness
        # alpha = contrast, beta = brightness
        out = cv2.convertScaleAbs(frame_bgr, alpha=float(self.cam_contrast), beta=int(self.cam_brightness))

        # Grayscale (convert back to BGR so downstream code (imshow/drawing) works)
        if self.cam_grayscale:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return out



    # ---------- main loop ----------

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # Decide what frame MediaPipe should see
            if self.cam_apply_to_tracking:
                frame_for_mp = self._apply_camera_adjustments(frame.copy())
            else:
                frame_for_mp = frame

            frame_rgb = cv2.cvtColor(frame_for_mp, cv2.COLOR_BGR2RGB)
            res = self.hands.process(frame_rgb)

            # Decide what you want to display
            frame_display = self._apply_camera_adjustments(frame.copy())


            pointer_hand = None
            action_hand = None

            # track best point gesture for auto
            best_point_conf = 0.0
            best_point_hand = None

            detected = []

            if res.multi_hand_landmarks and res.multi_handedness:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label  # "Left"/"Right"
                    # compute both normal + mirrored predictions, choose best
                    feat = landmarks_to_feature_vector(hand_lm, mirror=False)
                    pred1, conf1 = self.classifier.predict(feat)

                    feat_m = landmarks_to_feature_vector(hand_lm, mirror=True)
                    pred2, conf2 = self.classifier.predict(feat_m)

                    if conf2 > conf1:
                        pred, conf = pred2, conf2
                    else:
                        pred, conf = pred1, conf1

                    if conf < GESTURE_CONF_THRESHOLD:
                        pred = "none"
                        conf = 0.0

                    detected.append((label, hand_lm, pred, conf))

                    if pred == "point" and conf > best_point_conf:
                        best_point_conf = conf
                        best_point_hand = (label, hand_lm, pred, conf)

            # decide pointer vs action based on mode
            action_hands = []  # list of tuples (label, hand_lm, pred, conf)

            if self.hand_mode == "multi_keyboard":
                # BOTH hands become action hands (simultaneous keyboard)
                action_hands = detected[:]  # all detected hands are action sources
                pointer_hand = None         # no pointer hand in this mode by default

            else:
                # existing single pointer/action logic
                if self.hand_mode == "right":
                    for (label, hand_lm, pred, conf) in detected:
                        if label == "Right":
                            pointer_hand = (label, hand_lm, pred, conf)
                        elif label == "Left":
                            action_hand = (label, hand_lm, pred, conf)

                elif self.hand_mode == "left":
                    for (label, hand_lm, pred, conf) in detected:
                        if label == "Left":
                            pointer_hand = (label, hand_lm, pred, conf)
                        elif label == "Right":
                            action_hand = (label, hand_lm, pred, conf)

                else:  # auto
                    if best_point_hand is not None:
                        pointer_hand = best_point_hand
                        # action is the other hand if present
                        for item in detected:
                            if item is not pointer_hand:
                                action_hand = item
                                break
                    else:
                        # fallback: first is pointer, second is action
                        if len(detected) >= 1:
                            pointer_hand = detected[0]
                        if len(detected) >= 2:
                            action_hand = detected[1]

                # convert single action_hand to list
                if action_hand:
                    action_hands = [action_hand]
                else:
                    action_hands = []

            # ---- pointer movement ----
            if pointer_hand:
                _, hand_lm, _, _ = pointer_hand
                tip = hand_lm.landmark[8]  # index fingertip

                if self.mouse_mode == "DISABLED":
                    pass
                elif self.mouse_mode == "CURSOR":
                    self._cursor_move(tip.x, tip.y)
                else:
                    dx = (tip.x - 0.5) * self._joy_gain
                    dy = (tip.y - 0.5) * self._joy_gain
                    dx = self._apply_deadzone(dx)
                    dy = self._apply_deadzone(dy)
                    self._camera_move(dx, dy)

                # draw red dot on fingertip
                h, w = frame_display.shape[:2]
                cx, cy = int(tip.x * w), int(tip.y * h)
                cv2.circle(frame_display, (cx, cy), 8, (0, 0, 255), -1)

            # ---- action execution (MULTI-HAND for multi_keyboard) ----
            # Stop holds for hands that disappeared
            seen_labels = set([lbl for (lbl, _, _, _) in action_hands])

            for lbl in ["Left", "Right"]:
                if lbl not in seen_labels:
                    prev_hold = self.prev_hold_action_by_hand.get(lbl)
                    if prev_hold is not None:
                        prev_hold.stopHold()
                    self.prev_hold_action_by_hand[lbl] = None
                    self.prev_gesture_by_hand[lbl] = "none"

            mapped_texts = []
            fired_texts = []

            for (label, _, pred, conf) in action_hands:
                current_gesture = pred if conf >= GESTURE_CONF_THRESHOLD else "none"
                mapped_text, fired_text = self._process_action_for_hand(label, current_gesture)

                mapped_texts.append(f"{label}: {mapped_text}")
                fired_texts.append(f"{label}: {fired_text}")

            # UI: show both hands in multi_keyboard, else show single current gesture
            if self.hand_mode == "multi_keyboard":
                gL = self.prev_gesture_by_hand.get("Left", "none")
                gR = self.prev_gesture_by_hand.get("Right", "none")
                gesture_text = f"Gesture: L={gL} | R={gR}"
            else:
                # for normal modes, show the single action hand gesture
                # (if no action hand, it'll be "none")
                only = "none"
                if action_hands:
                    only = self.prev_gesture_by_hand.get(action_hands[0][0], "none")
                gesture_text = f"Gesture: {only}"

            mapped_text = " | ".join(mapped_texts) if mapped_texts else "None"
            fired_text = " | ".join(fired_texts) if fired_texts else "NO"

            mon_txt = "All Screens"
            if self.monitor:
                mon_txt = f"{self.screen_w}x{self.screen_h}"

            mode_text = f"Hand Mode: {self.hand_mode} | Mouse Mode: {self.mouse_mode} | Monitor: {mon_txt}"

            # Thread-safe UI update
            try:
                self.gui.root.after(
                    0,
                    self.gui.update_status,
                    mode_text,
                    gesture_text,
                    f"Mapped: {mapped_text}",
                    f"Fired: {fired_text}"
                )
            except Exception:
                pass



            cv2.imshow("Gesture Controller", frame_display)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                self.running = False
            if k == ord('r'):
                self.reload_profile_actions()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = GestureControllerApp()

    # Run OpenCV loop in a background thread so Tkinter can stay responsive
    t = threading.Thread(target=app.run, daemon=True)
    t.start()

    # Start GUI (main thread)
    app.gui.root.mainloop()

    # When GUI exits, stop the camera loop
    app.running = False
