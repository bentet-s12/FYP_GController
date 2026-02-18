import os
import time
import json
import shutil
import numpy as np
import cv2
import queue
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import ctypes
from ctypes import wintypes
import pyautogui  # Cursor mode absolute positioning
import socket
import threading
#from screeninfo import get_monitors
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
# --- PySide6 camera window (replaces cv2.imshow + cv2 trackbars) ---
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QShortcut , QKeySequence
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox, QFrame
)

from Actions import Actions  # your pydirectinput-based Actions.py
# ================== PATH CONFIG ==================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")
MODEL_TASK_PATH = os.path.join(SCRIPT_DIR, "data", "models", "hand_landmarker.task")
PROFILE_MANAGER_PATH = os.path.join(SCRIPT_DIR, "profileManager.json")

# IMPORTANT: this must match your real file name
PROFILE_JSON_PATH = os.path.join(SCRIPT_DIR, "Default.json")
GESTURELIST_JSON_PATH = os.path.join(SCRIPT_DIR, "GestureList.json")
STRICT_GESTURELIST = True  # if True: ignore profile mappings whose gesture is not in GestureList.json
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
CLASSES_PATH = os.path.join(DATA_DIR, "class_names.npy")
# ================== CONSTANTS ====================

K_NEIGHBORS = 3
GESTURE_CONF_THRESHOLD = 0.6
# Gestures where LEFT/RIGHT direction matters
DIRECTIONAL_GESTURES = {
    "thumbs_left",
    "thumbs_right",
}
SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001



SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040

# Win32 SendInput for relative mouse movement (CAMERA mode)
user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND

user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL

user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL

user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint
]
user32.SetWindowPos.restype = wintypes.BOOL


def bring_window_to_front(window_title: str) -> bool:
    hwnd = user32.FindWindowW(None, window_title)
    if not hwnd:
        return False

    user32.ShowWindow(hwnd, SW_RESTORE)

    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)

    user32.SetForegroundWindow(hwnd)
    return True

def load_profile_manager_json_backend():
    try:
        if not os.path.exists(PROFILE_MANAGER_PATH):
            return {}

        with open(PROFILE_MANAGER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception as e:
        print("[BACKEND] Failed to load profileManager.json:", e)
        return {}

def load_gesture_list(path=GESTURELIST_JSON_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_gesture_list(gestures, path=GESTURELIST_JSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(gestures), f, indent=4)

def gesturelist_add(name: str):
    g = load_gesture_list()
    if name not in g:
        g.append(name)
        save_gesture_list(g)

def gesturelist_remove(name: str):
    g = load_gesture_list()
    if name in g:
        g.remove(name)
        save_gesture_list(g)

def gesturelist_rename(old: str, new: str):
    g = load_gesture_list()
    if old in g:
        g = [new if x == old else x for x in g]
        g = list(dict.fromkeys(g))  # remove duplicates safely
        save_gesture_list(g)

def dataset_load():
    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH) and os.path.exists(CLASSES_PATH)):
        return None, None, None
    X = np.load(X_PATH, allow_pickle=False)
    y = np.load(Y_PATH, allow_pickle=True)
    classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
    print(f"[DATASET] Loaded: X={X.shape} y={len(y)} -> {X_PATH}", flush=True)
    return X, y, classes

def dataset_save(X, y, classes):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    np.save(CLASSES_PATH, np.array(classes, dtype=object))
    print(f"[DATASET] Saved: X={X.shape} y={len(y)} -> {X_PATH}", flush=True)

def dataset_delete_label(label: str) -> bool:
    label = (label or "").strip()
    if not label:
        return False

    X, y, classes = dataset_load()
    if X is None:
        print("[DELETE] Dataset files missing; nothing to do.")
        return False

    # normalize y to clean strings
    y_str = np.array([(str(v).strip()) for v in y], dtype=object)

    # also delete hand-suffixed variants if you use them
    targets = {label, f"{label}__L", f"{label}__R"}

    keep = np.array([v not in targets for v in y_str], dtype=bool)
    removed = int((~keep).sum())

    if removed == 0:
        print("[DELETE] No samples found for:", label, "targets=", targets)
        return False

    X2 = X[keep]
    y2 = y_str[keep]

    # rebuild classes from remaining labels
    classes2 = sorted(set(y2.tolist()))
    dataset_save(X2, y2, classes2)

    print(f"[DELETE] Removed {removed} samples for '{label}'. Remaining={len(y2)}")
    return True



def dataset_rename_label(old: str, new: str):
    X, y, classes = dataset_load()
    if X is None:
        return False
    y2 = np.array([new if v == old else v for v in y], dtype=y.dtype)
    classes2 = [new if c == old else c for c in classes]
    dataset_save(X, y2, classes2)
    return True

def landmarks_to_feature_vector(hand_lm, mirror=False):
    """
    Supports BOTH:
      - Solutions: hand_lm.landmark (21 pts)
      - Tasks:     hand_lm is a list of 21 landmarks (each has .x/.y)
    """
    # Tasks gives: list[NormalizedLandmark]
    if isinstance(hand_lm, list):
        pts = hand_lm
    else:
        # Solutions gives an object with .landmark
        pts = hand_lm.landmark

    coords = np.array([[p.x, p.y] for p in pts], dtype=np.float32)

    wrist = coords[0].copy()
    coords = coords - wrist

    scale = np.linalg.norm(coords[9]) + 1e-6
    coords = coords / scale

    if mirror:
        coords[:, 0] *= -1.0

    return coords.reshape(-1)


def load_actions_from_profile_json(profile_path: str):
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

            gesture = a.get("G_name")  # legacy fallback
            if not isinstance(gesture, str):
                continue
            gesture = gesture.strip()

            # skip empty/default placeholders
            if not gesture or gesture == "default":
                continue

            if STRICT_GESTURELIST and gesture_set and (gesture not in gesture_set):
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
            
            key_type = a.get("key_type")
            # DEFAULT if missing
            if not isinstance(key_type, str) or not key_type.strip():
                key_type = "Keyboard"
            else:
                kt = key_type.strip().lower()
                if kt == "mouse":
                    key_type = "Mouse"
                elif kt == "keyboard":
                    key_type = "Keyboard"

            # Build mapping that can be matched by BOTH:
            # - gesture binding (G_name)  e.g. "up", "down"
            # - action name (name)        e.g. "accelerate", "reverse"

            action_name = a.get("name")
            action_name = action_name.strip() if isinstance(action_name, str) else None

            # gesture variable already exists above (from G_name), but normalize "null"
            if gesture in ("", "null", "None"):
                gesture = None
            if action_name in ("", "null", "None"):
                action_name = None

            action_obj = Actions(
                name=action_name or (gesture or "unnamed"),
                G_name=gesture,
                key_pressed=key,
                input_type=input_type,
                key_type=key_type
            )

            # Store by gesture binding
            if gesture:
                action_map[gesture] = action_obj

            # Store by action name too
            if action_name:
                action_map[action_name] = action_obj


        print(f"[PROFILE] Loaded {len(action_map)} mappings from {os.path.basename(profile_path)}: {list(action_map.keys())}")
        return action_map

    except Exception as e:
        print("[PROFILE] Failed to load:", e)
        return action_map

def dataset_add_from_folder(label: str, folder: str):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".npy")]
    if not files:
        raise RuntimeError(f"No .npy samples found in {folder}")

    samples = [np.load(p, allow_pickle=False) for p in files]
    X_new = np.stack(samples, axis=0)
    y_new = np.array([label] * len(samples), dtype=object)

    X, y, classes = dataset_load()
    if X is None:
        X = np.empty((0, X_new.shape[1]), dtype=X_new.dtype)
        y = np.empty((0,), dtype=object)
        classes = []

    # normalize y to object strings (consistent)
    y = np.array([str(v) for v in y], dtype=object)

    X2 = np.vstack([X, X_new])
    y2 = np.concatenate([y, y_new])

    if label not in classes:
        classes.append(label)

    dataset_save(X2, y2, classes)

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

def normalize_handedness(label: str, flipped_for_detection: bool) -> str:
    if label not in ("Left", "Right"):
        return label
    # For your code: you DO flip before detection, so swap.
    if flipped_for_detection:
        return "Right" if label == "Left" else "Left"
    return label


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




class CameraWindow(QWidget):
    """
    Shows frames with preserved aspect ratio (letterbox) + sliders/checkbox.
    """
    def __init__(self, app_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app_ref  # GestureControllerApp instance (read/write cam params)

        self.setWindowTitle("Gesture Controller Camera")
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background: #030013;")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Frame display
        self.video = QLabel("No frame")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background: black; border-radius: 8px;")
        self.video.setMinimumHeight(300)
        root.addWidget(self.video, stretch=1)

        # Controls container (style it like your settings GUI)
        controls = QFrame()
        controls.setStyleSheet("background: #252438; border-radius: 12px;")
        c = QVBoxLayout(controls)
        c.setContentsMargins(12, 20, 12, 20)
        c.setSpacing(30)

        # ---- Detected Gesture Display ----
        # ---- Detected Gesture Text ----
        self.lbl_detected = QLabel("Detected: nothing")
        det_font = QFont()
        det_font.setPointSize(18)
        det_font.setBold(True)

        self.lbl_detected.setFont(det_font)
        self.lbl_detected.setAlignment(Qt.AlignCenter)
        self.lbl_detected.setStyleSheet("color: white; background: #252438; padding: 6px; border-radius: 8px;")

        root.addWidget(self.lbl_detected)


        # Contrast slider: map UI 0..350 => contrast -0.5..3.0
        row1 = QHBoxLayout()
        row1_label = QLabel("Contrast: ")
        font = row1_label.font()
        font.setPointSize(16)
        row1_label.setFont(font)
        row1.addWidget(row1_label)
        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setRange(0, 350)
        self.sld_contrast.setValue(int(round((float(self.app.cam_contrast) + 0.5) * 100)))
        self.sld_contrast.setStyleSheet("""
                                            background: #252438;
                                        """)
        row1.addWidget(self.sld_contrast, stretch=1)
        c.addLayout(row1)

        # Brightness slider: UI 0..200 => brightness -100..100
        row2 = QHBoxLayout()
        row2_label = QLabel("Brightness: ")
        font = row2_label.font()
        font.setPointSize(16)
        row2_label.setFont(font)
        row2.addWidget(row2_label)
        self.sld_brightness = QSlider(Qt.Horizontal)
        self.sld_brightness.setRange(0, 200)
        self.sld_brightness.setValue(int(self.app.cam_brightness) + 100)
        self.sld_brightness.setStyleSheet("""
                                            background: #252438;
                                        """)
        row2.addWidget(self.sld_brightness, stretch=1)
        c.addLayout(row2)

        # Greyscale checkbox (you asked: greyscale should be a checkbox)
        font = QFont()
        font.setPointSize(16)
        self.chk_gray = QCheckBox("Greyscale")
        self.chk_gray.setChecked(bool(self.app.cam_grayscale))
        self.chk_gray.setFont(font)
        self.chk_gray.setStyleSheet("""
                                        background: #252438;
                                        """)
        c.addWidget(self.chk_gray)

        # Optional: Apply-to-tracking checkbox (you already have it)
        self.chk_apply_tracking = QCheckBox("Apply adjustments to tracking (advanced)")
        self.chk_apply_tracking.setChecked(bool(self.app.cam_apply_to_tracking))
        self.chk_apply_tracking.setFont(font)
        self.chk_apply_tracking.setStyleSheet("""
                                            background: #252438;
                                        """)
        c.addWidget(self.chk_apply_tracking)

        # Basic label style (match your palette)
        controls.setStyleSheet("""
            QFrame { background: #252438; border-radius: 12px; }
            QLabel { color: #e0dde5; }
            QCheckBox { color: #e0dde5; }
        """)
        root.addWidget(controls, stretch=0)

        # Wire UI -> app variables
        self.sld_contrast.valueChanged.connect(self._on_contrast)
        self.sld_brightness.valueChanged.connect(self._on_brightness)
        self.chk_gray.toggled.connect(self._on_gray)
        self.chk_apply_tracking.toggled.connect(self._on_apply_tracking)

        self._last_pixmap = None
        self._last_key = None
        self.setFocusPolicy(Qt.StrongFocus)

    def set_detected_gesture_text(self, text: str):
        if not text:
            text = "nothing"
        self.lbl_detected.setText(f"Detected: {text}")


    def _on_contrast(self, v: int):
        self.app.cam_contrast = (float(v) / 100.0) - 0.50

    def _on_brightness(self, v: int):
        self.app.cam_brightness = int(v) - 100

    def _on_gray(self, on: bool):
        self.app.cam_grayscale = bool(on)

    def _on_apply_tracking(self, on: bool):
        self.app.cam_apply_to_tracking = bool(on)

    def set_frame_bgr(self, frame_bgr):
        """
        Convert BGR numpy frame -> QPixmap and display letterboxed with aspect ratio preserved.
        """
        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Build QImage (deep copy to be safe)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._last_pixmap = pix

        # Letterbox: keep aspect ratio inside label
        target = self.video.size()
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # re-apply scaling on resize
        if self._last_pixmap is not None:
            target = self.video.size()
            scaled = self._last_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video.setPixmap(scaled)

    def keyPressEvent(self, event):
        # store ASCII-ish keys: Space/Q/S/D/N etc.
        k = event.key()
        if k == Qt.Key_Space:
            self._last_key = 32
        else:
            txt = (event.text() or "")
            if txt:
                self._last_key = ord(txt.lower())
        event.accept()

    def pop_last_key(self):
        k = self._last_key
        self._last_key = None
        return k


# =================================================
#   GUI
# =================================================

class ClickTesterGUI:
    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.title("Gesture Controller Tester")
        self.root.geometry("780x620")
        self.root.resizable(True, True)

        self.main = tk.Frame(self.root)
        self.main.pack(fill="both", expand=True, padx=10, pady=10)

        self.click_count = 0
        self.label_mode = tk.Label(self.main, text="Hand Mode: MultiKB | Mouse Mode: CAMERA", font=("Arial", 12))
        self.label_mode.pack(pady=8)

        self.label_gesture = tk.Label(self.main, text="Gesture: none", font=("Arial", 14))
        self.label_gesture.pack(pady=8)

        self.label_mapped = tk.Label(self.main, text="Mapped: None", font=("Arial", 12))
        self.label_mapped.pack(pady=8)

        self.label_fired = tk.Label(self.main, text="Fired: NO", font=("Arial", 12))
        self.label_fired.pack(pady=8)

        # ---- Shortcuts help ----
        shortcuts = (
            "Shortcuts (works while GUI is focused):\n"
            "P = Cycle Hand Mode | R = Reload Profile | V = Toggle Vectors | C = Toggle Camera | Q = Quit"
        )
        self.label_shortcuts = tk.Label(self.main, text=shortcuts, font=("Arial", 11), justify="left")
        self.label_shortcuts.pack(pady=10)


        self.btn_reload = tk.Button(self.main, text="Reload Profile (r)", command=self.app.reload_profile_actions)
        self.btn_reload.pack(pady=10)

        self.btn_cam_toggle = tk.Button(
            self.main,
            text="Hide Camera View",
            command=self._toggle_camera_button
        )
        self.btn_cam_toggle.pack(pady=6)

        self.btn_vec_toggle = tk.Button(
            self.main,
            text="Hide Hand Vectors",
            command=self._toggle_vectors_button
        )
        self.btn_vec_toggle.pack(pady=6)



        # Hand mode buttons
        frame_hand = tk.Frame(self.main)
        frame_hand.pack(fill="x", expand=True, pady=6)
        tk.Label(frame_hand, text="Hand Mode: ").pack(side=tk.LEFT)
        tk.Button(frame_hand, text="Right pointer", command=lambda: self.app.set_hand_mode("right")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="Left pointer", command=lambda: self.app.set_hand_mode("left")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="Auto", command=lambda: self.app.set_hand_mode("auto")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_hand, text="MultiKB", command=lambda: self.app.set_hand_mode("multi_keyboard")).pack(side=tk.LEFT, padx=4)

        # Mouse mode buttons
        frame_mouse = tk.Frame(self.main)
        frame_mouse.pack(fill="x", expand=True, pady=6)
        tk.Label(frame_mouse, text="Mouse Mode: ").pack(side=tk.LEFT)
        tk.Button(frame_mouse, text="Disabled", command=lambda: self.app.set_mouse_mode("DISABLED")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mouse, text="Camera", command=lambda: self.app.set_mouse_mode("CAMERA")).pack(side=tk.LEFT, padx=4)
        tk.Button(frame_mouse, text="Cursor", command=lambda: self.app.set_mouse_mode("CURSOR")).pack(side=tk.LEFT, padx=4)

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

        # Brightness slider
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

        # Apply to tracking toggle
        self.track_adj_var = tk.BooleanVar(value=self.app.cam_apply_to_tracking)
        tk.Checkbutton(
            frame_cam, text="Apply adjustments to tracking (advanced)",
            variable=self.track_adj_var,
            command=self._on_track_adj_toggle
        ).grid(row=2, column=1, sticky="w", pady=(4,0))

        frame_cam.columnconfigure(1, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

        # Bind keys globally within this Tk window (works even when buttons/sliders are focused)
        self.root.bind_all("<KeyPress>", self._on_keypress)
        # Start hidden – only shown when toggled
        self.root.withdraw()

    def on_close(self):
        self.running = False
        self.app.running = False
        self.app.want_camera_view = False
        try:
            self.root.after(50, self.root.destroy)
        except Exception:
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

    def _toggle_camera_button(self):
        self.app.toggle_camera_view()
        self.btn_cam_toggle.config(
            text="Hide Camera View" if self.app.want_camera_view else "Show Camera View"
        )

    def _toggle_vectors_button(self):
        self.app.toggle_hand_vectors()
        self.btn_vec_toggle.config(
            text="Hide Hand Vectors" if self.app.show_hand_vectors else "Show Hand Vectors"
        )

    def _on_keypress(self, event):
        k = (event.keysym or "").lower()

        if k == "p":
            self.app.cycle_hand_mode()

        elif k == "r":
            self.app.reload_profile_actions()

        elif k == "v":
            self.app.toggle_hand_vectors()
            # If you have a vectors button, keep its text in sync:
            if hasattr(self, "btn_vec_toggle"):
                self.btn_vec_toggle.config(
                    text="Hide Hand Vectors" if self.app.show_hand_vectors else "Show Hand Vectors"
                )

        elif k == "c":
            self.app.toggle_camera_view()
            if hasattr(self, "btn_cam_toggle"):
                self.btn_cam_toggle.config(
                    text="Hide Camera View" if self.app.want_camera_view else "Show Camera View"
                )

        elif k == "q":
            # exit to background (do NOT kill backend)
            self.app._exit_collect_to_background()
            try:
                self.root.after(0, self.hide)
            except Exception:
                pass

    def hide(self):
        try:
            self.root.withdraw()
        except Exception:
            pass


    def show(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

            # reliable Windows raise
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
            self.root.attributes("-topmost", False)
        except Exception:
            pass


    def is_visible(self):
        try:
            return self.root.state() != "withdrawn"
        except Exception:
            return True

    def bring_to_front(self):
        """Bring the Tk window to the front (Windows-friendly)."""
        try:
            self.root.deiconify()      # un-minimize if minimized
            self.root.lift()           # raise above other windows
            self.root.focus_force()    # take focus (may be limited by Windows focus rules)

            # Windows: briefly set topmost then revert (works very reliably)
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
            self.root.attributes("-topmost", False)
        except Exception:
            pass

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass


'''
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
'''
class CommandServer(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=50555):
        super().__init__(daemon=True)
        self.app = app
        self.host = host
        self.port = port
        self._stop_flag = False

    def run(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(5)
            s.settimeout(0.5)  # so thread can check stop flag periodically
            print(f"[CMD] Listening on {self.host}:{self.port}", flush=True)
        except Exception as e:
            print(f"[CMD] FAILED to start on {self.host}:{self.port} -> {e}", flush=True)
            return

        with s:
            while not self._stop_flag:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[CMD] accept() error: {e}", flush=True)
                    continue

                with conn:
                    try:
                        conn.settimeout(1.0)
                        data = conn.recv(1024).decode("utf-8", errors="ignore").strip()
                    except Exception as e:
                        print(f"[CMD] recv error: {e}", flush=True)
                        continue

                    print(f"[CMD] Received: {data}", flush=True)

                    try:
                        if data == "PING":
                            conn.sendall(b"PONG\n")

                        elif data == "TOGGLE_CAMERA":
                            self.app._req_toggle_camera = True
                            conn.sendall(b"OK\n")

                        elif data == "TOGGLE_GUI":
                            self.app._req_toggle_gui = True
                            conn.sendall(b"OK\n")

                        elif data == "QUIT":
                            self.app.running = False
                            self._stop_flag = True
                            conn.sendall(b"OK\n")

                        elif data.startswith("CREATE_GESTURE"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2 or not parts[1].strip():
                                conn.sendall(b"ERR Empty gesture name\n")
                            else:
                                self.app._collect_req_name = parts[1].strip()
                                conn.sendall(b"OK\n")

                        elif data.startswith("DELETE_GESTURE"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2 or not parts[1].strip():
                                conn.sendall(b"ERR Empty gesture name\n")
                            else:
                                self.app._delete_req_name = parts[1].strip()
                                conn.sendall(b"OK\n")

                        elif data.startswith("SET_PROFILE"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2 or not parts[1].strip():
                                conn.sendall(b"ERR Empty profile id\n")
                            else:
                                ok = self.app.set_active_profile(parts[1].strip())
                                conn.sendall(b"OK\n" if ok else b"ERR Failed\n")
                                
                        elif data == "GET_MODES":
                            hand = getattr(self.app, "hand_mode", "auto")
                            mouse = getattr(self.app, "mouse_mode", "DISABLED")
                            conn.sendall(f"OK {hand} {mouse}\n".encode("utf-8"))

                        elif data == "CYCLE_HAND_MODE":
                            self.app.cycle_hand_mode()
                            conn.sendall(b"OK\n")

                        elif data == "CYCLE_MOUSE_MODE":
                            self.app.cycle_mouse_mode()
                            conn.sendall(b"OK\n")

                        elif data == "TOGGLE_VECTORS":
                            try:
                                self.app.toggle_hand_vectors()
                                conn.sendall(b"OK\n")
                            except Exception as e:
                                conn.sendall(f"ERR {e}\n".encode("utf-8", errors="ignore"))

                        elif data == "RELOAD_PROFILE":
                            try:
                                self.app.reload_profile_actions()
                                conn.sendall(b"OK\n")
                            except Exception as e:
                                conn.sendall(f"ERR {e}\n".encode("utf-8", errors="ignore"))

                        elif data == "GET_HAND_MODE":
                            # return something like: right / left / auto / multi_keyboard
                            conn.sendall((f"OK {self.app.hand_mode}\n").encode("utf-8"))

                        elif data == "GET_MOUSE_MODE":
                            # return something like: DISABLED / CAMERA / CURSOR
                            conn.sendall((f"OK {self.app.mouse_mode}\n").encode("utf-8"))

                        elif data.startswith("SET_HAND_MODE"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2:
                                conn.sendall(b"ERR Missing hand mode\n")
                            else:
                                ok = self.app.set_hand_mode(parts[1].strip())
                                conn.sendall(b"OK\n" if ok else b"ERR Bad mode\n")

                        elif data.startswith("SET_MOUSE_MODE"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2:
                                conn.sendall(b"ERR Missing mouse mode\n")
                            else:
                                ok = self.app.set_mouse_mode(parts[1].strip())
                                conn.sendall(b"OK\n" if ok else b"ERR Bad mode\n")

                        elif data.startswith("SET_VECTORS"):
                            parts = data.split(maxsplit=1)
                            if len(parts) < 2:
                                conn.sendall(b"ERR Missing vectors state\n")
                            else:
                                ok = self.app.set_vectors_visible(parts[1].strip())
                                conn.sendall(b"OK\n" if ok else b"ERR Bad value\n")
                        else:
                            conn.sendall(b"UNKNOWN\n")

                    except Exception as e:
                        print(f"[CMD] send error: {e}", flush=True)



# =================================================
#   MAIN APP
# =================================================

class GestureControllerApp:
    def __init__(self, enable_gui: bool = True, enable_camera: bool = True):
        self.running = True
        self.enable_gui = enable_gui
        self.enable_camera = enable_camera
        self._req_toggle_gui = False
        self.want_gui = False          # start hidden
        self._gui_applied = None       # tracks last applied state
        self._raised_once = False
        self._gui_queue = queue.Queue()
        self._gui_thread = None
        self.gui = None  # created inside GUI thread
        self._mode_cooldown = 1.0   # seconds
        self._last_hand_mode_change = 0.0
        self._last_mouse_mode_change = 0.0

        self.want_camera_view = False
        self._window_name = "Gesture Controller"
        self._req_toggle_camera = False

        self.hand_mode = "multi_keyboard"
        self.mouse_mode = "DISABLED"
        self.show_hand_vectors = True

        self.hand_mode_cycle = ["multi_keyboard","right", "left","auto"]
        self._hand_mode_idx = self.hand_mode_cycle.index(self.hand_mode)
        self.mouse_mode_cycle = ["DISABLED", "CAMERA", "CURSOR"]
        self._mouse_mode_idx = self.mouse_mode_cycle.index(self.mouse_mode)

        self.swap_handedness = False

        self.cam_contrast = 1.0
        self.cam_brightness = 0
        self.cam_grayscale = False
        self.cam_apply_to_tracking = False

        self._joy_gain = 70.0
        self._joy_alpha = 0.18
        self._joy_deadzone = 0.04
        self._joy_max_step = 70.0
        self._joy_sm_x = 0.0
        self._joy_sm_y = 0.0

        self.action_map = load_actions_from_profile_json(PROFILE_JSON_PATH)
        self.prev_gesture_by_hand = {"Left": "none", "Right": "none"}
        self.prev_hold_action_by_hand = {"Left": None, "Right": None}
        self.MULTI_KEYBOARD_ONLY = True

        self.collect_mode = False
        self.collect_name = None
        self.collect_running = False
        self.collect_paused = True   # SPACE toggles
        self.collect_cancelled = False
        self.collect_done = False

        self.collect_target = 200
        self.collect_saved = 0
        self.collect_interval = 0.1
        self._collect_last_save_t = 0.0

        self._collect_req_name = None
        self._collect_status = "IDLE"   # IDLE / RUNNING / DONE / CANCELLED / ERR:...

        self.cap = None
        self.hand_landmarker = None
        self.classifier = None
        self._mp_start_t = time.perf_counter()

        if self.enable_gui:
            self._start_gui_thread()

        # --- collector mode ---
        self.collect_mode = "TWO_HAND"     # "TWO_HAND" or "ONE_HAND"
        self.collect_phase = "LEFT"        # only used in TWO_HAND (LEFT then RIGHT)
        self.collect_target_one = "LEFT"   # only used in ONE_HAND (LEFT/RIGHT)
        self.collect_saved_phase = {"LEFT": 0, "RIGHT": 0}
        self._key_latch = {"q": False, "space": False, "s": False, "d": False}

        self.collect_auto_pause_on_lost = True
        self.collect_waiting_for_hand = False
        self.collect_locked_label = None  # "Left"/"Right" once capture starts


        self.suspend_actions = False
        self._prev_mouse_mode = None
        self._prev_hand_mode = None

        self._gui_toggle_pending = False
        self._gui_target_visible = False  # last requested state

        self._last_tk_pump = 0.0
        self._tk_pump_interval = 1.0 / 60.0   # 60 Hz (use 1/30 if you want)

        self._delete_req_name = None
        self._req_profile_id = None
        self.current_profile_path = PROFILE_JSON_PATH

        self.qt_app = None
        self.camera_qt = None
        self._last_frame_for_qt = None



        if self.enable_camera:
            self._init_camera_and_models()
            self._init_monitor()
        
        if self.gui is not None:
            self.gui.hide()

    def _start_gui_thread(self):
        self._gui_thread = threading.Thread(target=self._gui_thread_main, daemon=True)
        self._gui_thread.start()

    def _gui_thread_main(self):
        # Tk must be CREATED in the same thread that runs mainloop
        try:
            self.gui = ClickTesterGUI(self)
            self.gui.hide()  # start hidden

            def pump_queue():
                # Process all pending GUI messages
                try:
                    while True:
                        msg = self._gui_queue.get_nowait()
                        if msg is None:
                            # shutdown signal
                            try:
                                self.gui.destroy()
                            except Exception:
                                pass
                            return

                        kind = msg[0]

                        if kind == "SHOW":
                            self.gui.show()

                        elif kind == "HIDE":
                            self.gui.hide()

                        elif kind == "STATUS":
                            _, mode_text, gesture_text, mapped_text, fired_text = msg
                            self.gui.update_status(mode_text, gesture_text, mapped_text, fired_text)

                except queue.Empty:
                    pass
                except Exception as e:
                    print("[GUI] pump_queue error:", e, flush=True)

                # reschedule
                try:
                    self.gui.root.after(16, pump_queue)  # ~60fps GUI responsiveness
                except Exception:
                    pass

            self.gui.root.after(16, pump_queue)
            self.gui.root.mainloop()

        except Exception as e:
            print("[GUI] thread crashed:", e, flush=True)
            self.gui = None

    def request_collect_gesture(self, name: str):
        name = (name or "").strip()
        if not name:
            return False, "Empty gesture name"
        if self.collect_running:
            return False, "Collector already running"
        self._collect_req_name = name
        self._collect_status = "STARTING"
        return True, "Starting"

    def get_collect_status(self) -> str:
        return self._collect_status
    
    def _init_monitor(self):
        # if don't need cursor mode immediately, can skip prompting
        # self.monitor = None
        screen_w, screen_h = pyautogui.size()
        self.screen_x = 0
        self.screen_y = 0
        self.screen_w = screen_w
        self.screen_h = screen_h
    
    def _ensure_qt_camera(self):
        if self.camera_qt is not None:
            return

        # Create QApplication only once
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            self.qt_app = QApplication([])
        else:
            self.qt_app = QApplication.instance()

        self.camera_qt = CameraWindow(self)
        self.camera_qt.hide()
        self._install_camera_shortcuts_from_pm()

    def _init_camera_and_models(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("[CAM] Failed to open camera.")

        self.classifier = KNNGestureClassifier()
        self.classifier.load_dataset()

        if not os.path.isfile(MODEL_TASK_PATH):
            raise FileNotFoundError(f"Missing model file: {MODEL_TASK_PATH}")

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp_start_t = time.perf_counter()

    def set_hand_mode(self, mode: str) -> bool:
        now = time.perf_counter()
        if now - self._last_hand_mode_change < self._mode_cooldown:
            print("[MODE] Cooldown active (hand).")
            return False
        mode = (mode or "").strip().lower()
        
        mapping = {
            "right": "right",
            "left": "left",
            "auto": "auto",
            "multikb": "multi_keyboard",
            "multi_keyboard": "multi_keyboard",
        }
        if mode not in mapping:
            print("[MODE] Bad hand_mode:", mode, flush=True)
            return False
        self._last_hand_mode_change = now
        self.hand_mode = mapping[mode]
        # keep cycle index consistent
        if self.hand_mode in self.hand_mode_cycle:
            self._hand_mode_idx = self.hand_mode_cycle.index(self.hand_mode)

        print("[MODE] hand_mode ->", self.hand_mode, flush=True)
        return True

    def set_mouse_mode(self, mode: str) -> bool:
        mode = (mode or "").strip().upper()

        if mode not in self.mouse_mode_cycle:
            print("[MODE] Bad mouse_mode:", mode, flush=True)
            return False

        self.mouse_mode = mode
        self._mouse_mode_idx = self.mouse_mode_cycle.index(mode)

        if mode == "DISABLED":
            self._reset_mouse_state()

        print("[MODE] mouse_mode ->", self.mouse_mode, flush=True)
        return True



    
    def set_vectors_visible(self, val: str) -> bool:
        v = (val or "").strip().lower()
        if v in ("on", "1", "true", "yes"):
            self.show_hand_vectors = True
            return True
        if v in ("off", "0", "false", "no"):
            self.show_hand_vectors = False
            return True
        return False
    
    def _hold_token(self, act):
        kt = act.getKeyType() or "Keyboard"
        key = act.getKeyPressed()
        return (kt, key)

    def _hold_down(self, token):
        kt, key = token
        if not key:
            return

        # init store
        if not hasattr(self, "_held_counts"):
            self._held_counts = {}

        c = self._held_counts.get(token, 0) + 1
        self._held_counts[token] = c

        if c == 1:
            # first holder -> press down
            try:
                if kt == "Mouse":
                    import pydirectinput
                    pydirectinput.mouseDown(button=key)
                else:
                    import pydirectinput
                    pydirectinput.keyDown(key)
            except Exception as e:
                print("[HOLD] down failed:", token, e)

    def _hold_up(self, token):
        kt, key = token
        if not key or not hasattr(self, "_held_counts"):
            return

        c = self._held_counts.get(token, 0)
        if c <= 0:
            return

        c -= 1
        if c == 0:
            # last holder -> release
            try:
                if kt == "Mouse":
                    import pydirectinput
                    pydirectinput.mouseUp(button=key)
                else:
                    import pydirectinput
                    pydirectinput.keyUp(key)
            except Exception as e:
                print("[HOLD] up failed:", token, e)
            self._held_counts.pop(token, None)
        else:
            self._held_counts[token] = c

    def reload_profile_actions(self):
        path = getattr(self, "active_profile_path", None) or getattr(self, "current_profile_path", None) or PROFILE_JSON_PATH
        print("[PROFILE] Reloading:", path, "exists=", os.path.exists(path), flush=True)

        if not os.path.exists(path):
            print("[PROFILE] Reload skipped (missing):", path, flush=True)
            return

        self.action_map = load_actions_from_profile_json(path)

    def set_active_profile(self, profile_id: str) -> bool:
        pid = (profile_id or "").strip()
        if not pid:
            return False

        # Accept: "1", "profile_1", "profile_1.json", "Default", "Default.json"
        if pid.lower() in ("default", "default.json"):
            path = os.path.join(SCRIPT_DIR, "Default.json")

        elif pid.lower().endswith(".json"):
            # treat as direct filename under src
            path = os.path.join(SCRIPT_DIR, pid)

        elif pid.startswith("profile_"):
            path = os.path.join(SCRIPT_DIR, f"{pid}.json")

        else:
            path = os.path.join(SCRIPT_DIR, f"profile_{pid}.json")

        print(f"[PROFILE] Switching to: {path} exists= {os.path.exists(path)}", flush=True)

        # If missing, don't switch (keep previous mappings)
        if not os.path.exists(path):
            print(f"[PROFILE] Missing: {path} (keeping current profile)", flush=True)
            return True

        self.active_profile_id = pid
        self.active_profile_path = path
        self.current_profile_path = path

        self.action_map = load_actions_from_profile_json(path)
        return True

    def request_collect_gesture(self, name: str):
        name = (name or "").strip()
        if not name:
            return False, "Empty gesture name"
        if self.collect_running:
            return False, "Collector already running"

        self._collect_req_name = name
        self._collect_status = "STARTING"
        return True, "Starting"

    def get_collect_status(self) -> str:
        return self._collect_status

    def _start_collect_mode(self, gesture_name: str):
        self.collect_running = True
        self.collect_name = gesture_name
        # FORCE camera to show immediately when collection begins
        self.want_camera_view = True

        # reset counters
        self.collect_saved = 0
        self.collect_saved_phase = {"LEFT": 0, "RIGHT": 0}
        self.collect_phase = "LEFT"

        # default mode
        self.collect_mode = "TWO_HAND"      # start in TWO_HAND
        self.collect_target_one = "LEFT"    # used only if ONE_HAND

        self.collect_paused = True          # SPACE to start
        self.collect_cancelled = False
        self.collect_done = False
        self._collect_last_save_t = 0.0
        self._collect_status = "RUNNING"

        print(f"[COLLECT] Ready: {gesture_name} | Mode={self.collect_mode} (S toggles) | SPACE start/pause | Q cancel")

        # stop actions from injecting keys during collect
        self.suspend_actions = True

        # freeze runtime modes during collect
        self._prev_mouse_mode = self.mouse_mode
        self._prev_hand_mode = self.hand_mode
        self.mouse_mode = "DISABLED"   # prevent cursor/mouse injection during collect

        # stop any holds immediately
        for lbl in ["Left", "Right"]:
            prev_hold = self.prev_hold_action_by_hand.get(lbl)
            if prev_hold is not None:
                try:
                    prev_hold.stopHold()
                except Exception:
                    pass
            self.prev_hold_action_by_hand[lbl] = None
            self.prev_gesture_by_hand[lbl] = "none"


    def _collect_step(self, detected):
        if self.collect_done or self.collect_cancelled:
            return

        want = self._collect_want_label()  # "Left"/"Right"

        # If paused, we still update waiting message but never save
        if self.collect_paused:
            self.collect_waiting_for_hand = False
            return

        # Once we start/resume, lock to wanted label if not locked yet
        if self.collect_locked_label is None:
            self.collect_locked_label = want

        desired = self.collect_locked_label

        # Find the correct hand only (never save wrong hand)
        picked = None
        for (label, hand_lm, pred, conf) in detected:
            if label.lower() == desired.lower():
                picked = hand_lm
                break

        # If correct hand not found: auto-pause but DO NOT reset counts
        if picked is None:
            self.collect_waiting_for_hand = True
            if self.collect_auto_pause_on_lost:
                self.collect_paused = True
            return

        self.collect_waiting_for_hand = False

        # Throttle AFTER we have the correct hand
        now = time.perf_counter()
        if (now - self._collect_last_save_t) < self.collect_interval:
            return

        # Save feature vector
        vec = landmarks_to_feature_vector(picked, mirror=False)
        out_dir = os.path.join(SCRIPT_DIR, "data", "tmp_collect", self.collect_name)
        os.makedirs(out_dir, exist_ok=True)

        # Choose filename based on mode
        if self.collect_mode == "ONE_HAND":
            idx = self.collect_saved
            fname = f"{self.collect_name}_{desired}_{idx:04d}.npy"
        else:
            idx = self.collect_saved_phase["LEFT"] if self.collect_phase == "LEFT" else self.collect_saved_phase["RIGHT"]
            fname = f"{self.collect_name}_{self.collect_phase}_{idx:04d}.npy"

        np.save(os.path.join(out_dir, fname), vec)

        # Update counters
        self._collect_last_save_t = now

        if self.collect_mode == "ONE_HAND":
            self.collect_saved += 1
            if self.collect_saved >= self.collect_target:
                self.collect_done = True
                self.collect_running = False
                self._finish_collect_success()
        else:
            per = self.collect_target // 2
            if self.collect_phase == "LEFT":
                self.collect_saved_phase["LEFT"] += 1
                if self.collect_saved_phase["LEFT"] >= per:
                    self.collect_phase = "RIGHT"
                    self.collect_paused = True           # pause between phases
                    self.collect_locked_label = None     # relock for right
            else:
                self.collect_saved_phase["RIGHT"] += 1
                if self.collect_saved_phase["RIGHT"] >= per:
                    self.collect_done = True
                    self.collect_running = False
                    self._finish_collect_success()


    def _finish_collect_success(self):
        tmp_folder = os.path.join(SCRIPT_DIR, "data", "tmp_collect", self.collect_name)

        try:
            # 1) Register gesture name
            gesturelist_add(self.collect_name)

            # 2) Merge tmp_collect → main dataset
            if self.collect_mode == "ONE_HAND":
                suffix = "__R" if self.collect_target_one == "RIGHT" else "__L"
                label_to_save = self.collect_name + suffix
            else:
                label_to_save = self.collect_name

            dataset_add_from_folder(label_to_save, tmp_folder)


            # 3) Reload classifier
            if self.classifier:
                self.classifier.load_dataset()

            self._collect_status = "DONE"
            print(f"[COLLECT] DONE: {self.collect_name}")

        except Exception as e:
            self._collect_status = f"ERR: {e}"
            print("[COLLECT] ERROR:", e)
            return

        finally:
            # 4) CLEANUP tmp_collect
            import shutil
            shutil.rmtree(tmp_folder, ignore_errors=True)

            # 5) Restore runtime state
            self.collect_running = False
            self.collect_done = True
            self.suspend_actions = False

            if self._prev_mouse_mode is not None:
                self.mouse_mode = self._prev_mouse_mode
            if self._prev_hand_mode is not None:
                self.hand_mode = self._prev_hand_mode

            self.collect_locked_label = None

    def _add_null_gesture_sample(self):
        """
        Adds a single "null" gesture sample = 42D zero vector into the dataset,
        ensures GestureList.json contains "null", then reloads the classifier.
        """
        null_name = "null"

        # 1) Ensure GestureList has "null"
        try:
            gestures = load_gesture_list(GESTURELIST_JSON_PATH)
            if null_name not in gestures:
                gestures.append(null_name)
                with open(GESTURELIST_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(gestures, f, indent=4)
                print("[NULL] Added 'null' to GestureList.json")
        except Exception as e:
            print("[NULL] Failed updating GestureList.json:", e)

        # 2) Append a zero-vector sample to X/y (and update class_names)
        X_path = os.path.join(DATA_DIR, "X.npy")
        y_path = os.path.join(DATA_DIR, "y.npy")
        c_path = os.path.join(DATA_DIR, "class_names.npy")

        vec = np.zeros((42,), dtype=np.float32)

        try:
            if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(c_path):
                X = np.load(X_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)

                # force y to strings (matches your loader behavior)
                y = np.array([str(v) for v in y], dtype=object)

                # append
                X2 = np.vstack([X, vec.reshape(1, -1)]) if X.ndim == 2 else np.array(list(X) + [vec], dtype=object)
                y2 = np.append(y, null_name)

                class_names2 = sorted(set(list(y2)))

                np.save(X_path, X2)
                np.save(y_path, y2)
                np.save(c_path, np.array(class_names2, dtype=object))

                print(f"[NULL] Appended null sample. Total samples: {len(y)} -> {len(y2)}")
            else:
                print("[NULL] Dataset files missing (X/y/class_names). Cannot append null sample.")
                return

        except Exception as e:
            print("[NULL] Failed appending null sample:", e)
            return

        # 3) Reload classifier so it immediately recognizes "null"
        try:
            if self.classifier:
                self.classifier.load_dataset()
                print("[NULL] Classifier reloaded.")
        except Exception as e:
            print("[NULL] Classifier reload failed:", e)


    def _draw_collect_overlay(self, frame):
        mode = self.collect_mode
        want = self._collect_want_label()

        if mode == "ONE_HAND":
            prog = f"{self.collect_saved}/{self.collect_target}"
            extra = f"Target: {self.collect_target_one} (D toggles)"
        else:
            per = self.collect_target // 2
            prog = f"L {self.collect_saved_phase['LEFT']}/{per} | R {self.collect_saved_phase['RIGHT']}/{per}"
            extra = f"Phase: {self.collect_phase}"

        cv2.putText(frame, f"COLLECT MODE: {self.collect_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"Mode: {mode}  (S toggles)", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, extra, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Saved: {prog}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Need: {want}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

        cv2.putText(frame, "SPACE:start/pause | Q:cancel | S:mode | D:1/2 hand", (10, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

        cv2.putText(frame, "| N:null input",(10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

        if self.collect_paused:
            cv2.putText(frame, "PAUSED", (10, 275),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            
        if self.collect_waiting_for_hand and not self.collect_paused:
            cv2.putText(frame, "WAITING FOR HAND...", (10, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)


            
    def _collect_want_label(self) -> str:
        # MediaPipe labels are "Left"/"Right"
        if self.collect_mode == "ONE_HAND":
            return "Left" if self.collect_target_one == "LEFT" else "Right"
        else:
            return "Left" if self.collect_phase == "LEFT" else "Right"


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
        prev_g = self.prev_gesture_by_hand.get(hand_label, "none")
        prev_hold_token = self.prev_hold_action_by_hand.get(hand_label)  # now stores token tuple, not Actions

        act = self.action_map.get(current_gesture) if current_gesture and current_gesture != "none" else None

        # MultiKB keyboard-only restriction
        if self.hand_mode == "multi_keyboard" and self.MULTI_KEYBOARD_ONLY and act is not None:
            if (act.getKeyType() or "Keyboard") != "Keyboard":
                act = None

        # If gesture changed or disappeared, release previous hold for THIS hand only
        if prev_hold_token is not None:
            # If no longer holding same gesture/action, release
            if current_gesture == "none" or act is None:
                self._hold_up(prev_hold_token)
                prev_hold_token = None
            else:
                # if the new hold target differs from what we were holding, release old first
                new_token = self._hold_token(act)
                if new_token != prev_hold_token:
                    self._hold_up(prev_hold_token)
                    prev_hold_token = None

        fired_text = "NO"
        mapped_text = "None"

        if act is not None:
            input_type = act.getInputType()
            key = act.getKeyPressed()

            # normalize input_type
            if isinstance(input_type, str):
                t = input_type.strip().lower().replace(" ", "_")
                if t == "click":
                    input_type = "Click"
                elif t == "hold":
                    input_type = "Hold"
                elif t in ("d_click", "doubleclick", "double_click"):
                    input_type = "D_Click"

            mapped_text = f"{current_gesture} -> {key} ({input_type})"

            # edge trigger for click types
            is_new = (current_gesture != prev_g)

            if input_type in ("Click", "D_Click"):
                if is_new:
                    act.useAction(current_gesture)
                    fired_text = "YES"

            elif input_type == "Hold":
                if prev_hold_token is None:
                    token = self._hold_token(act)
                    self._hold_down(token)
                    prev_hold_token = token
                fired_text = "HOLD"

        # save state
        self.prev_gesture_by_hand[hand_label] = current_gesture
        self.prev_hold_action_by_hand[hand_label] = prev_hold_token

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
    
    def toggle_camera_view(self):
        self.want_camera_view = not self.want_camera_view

    def _install_camera_shortcuts_from_pm(self):
        from PySide6.QtGui import QShortcut, QKeySequence
        from PySide6.QtCore import Qt

        if self.camera_qt is None:
            return

        # Load profileManager.json using existing loader logic
        pm = load_profile_manager_json_backend()
        sc_map = pm.get("shortcuts", {}) if isinstance(pm, dict) else {}

        # Clear old shortcuts
        old = getattr(self, "_camera_shortcuts", None)
        if old:
            for s in old:
                try:
                    s.setEnabled(False)
                    s.deleteLater()
                except Exception:
                    pass

        self._camera_shortcuts = []

        def bind(action_name: str, callback):
            key_str = (sc_map.get(action_name) or "").strip()
            if not key_str:
                return
            s = QShortcut(QKeySequence(key_str), self.camera_qt)
            s.setContext(Qt.ApplicationShortcut)
            s.activated.connect(callback)
            self._camera_shortcuts.append(s)

        # Same actions used in main window
        bind("cycle_hand_mode", self.cycle_hand_mode)
        bind("cycle_mouse_mode", self.cycle_mouse_mode)
        bind("toggle_vectors", self.toggle_hand_vectors)
        bind("toggle_camera", self.toggle_camera_view)
        bind("reload_profile", self.reload_profile_actions)

    def toggle_hand_vectors(self):
        self.show_hand_vectors = not self.show_hand_vectors

    def normalize_handedness(self, label: str) -> str:
        if label not in ("Left", "Right"):
            return label
        if self.swap_handedness:
            return "Right" if label == "Left" else "Left"
        return label

    def cycle_hand_mode(self):
        now = time.perf_counter()

        if now - self._last_hand_mode_change < self._mode_cooldown:
            print("[MODE] Hand mode cooldown active.")
            return

        self._last_hand_mode_change = now

        self._hand_mode_idx = (self._hand_mode_idx + 1) % len(self.hand_mode_cycle)
        self.hand_mode = self.hand_mode_cycle[self._hand_mode_idx]

        print(f"[MODE] Hand mode switched to: {self.hand_mode}")


    def cycle_mouse_mode(self):
        now = time.perf_counter()
        if now - self._last_mouse_mode_change < self._mode_cooldown:
            print("[MODE] Cooldown active (mouse).")
            return

        self._last_mouse_mode_change = now

        # Advance index
        self._mouse_mode_idx = (self._mouse_mode_idx + 1) % len(self.mouse_mode_cycle)
        self.mouse_mode = self.mouse_mode_cycle[self._mouse_mode_idx]

        if self.mouse_mode == "DISABLED":
            self._reset_mouse_state()

        print(f"[MODE] Mouse mode switched to: {self.mouse_mode}", flush=True)




    def enforce_hand_suffix(self, pred_label: str, hand_label: str) -> str:
        """
        Enforce one-hand-only gestures.
        If a class was trained as gesture__L or gesture__R,
        only allow it on the corresponding hand.
        """
        if not pred_label or pred_label == "none":
            return pred_label

        if pred_label.endswith("__L"):
            base = pred_label[:-3]
            return base if hand_label == "Left" else "none"

        if pred_label.endswith("__R"):
            base = pred_label[:-3]
            return base if hand_label == "Right" else "none"

        return pred_label

    def start_camera_thread(self):
        import threading
        self._cam_thread = threading.Thread(target=self.run, daemon=True)
        self._cam_thread.start()

    def stop(self):
        self.running = False
        # release resources safely from run() cleanup

    def set_camera_visible(self, visible: bool):
        self.want_camera_view = bool(visible)

    def toggle_camera_visible(self):
        self.want_camera_view = not self.want_camera_view

    def toggle_clicker_gui(self):
        if self.gui is None:
            return
        # Tk calls must be scheduled on Tk thread
        def _do():
            if self.gui.is_visible():
                self.gui.hide()
            else:
                self.gui.show()
        try:
            self.gui.root.after(0, _do)
        except Exception:
            pass
    
    def _cancel_collection(self, reason="CANCELLED"):
        self.collect_cancelled = True
        self.collect_running = False
        self.collect_paused = True
        self._collect_status = reason
        self.collect_locked_label = None
        print(f"[COLLECT] {reason}")

    def _cancel_collect_and_cleanup(self):
        # mark mode exit
        self.collect_cancelled = True
        self.collect_done = False
        self.collect_paused = True

        # cleanup tmp folder for this gesture (only if it exists)
        if self.collect_name:
            tmp = os.path.join(SCRIPT_DIR, "data", "tmp_collect", self.collect_name)
            shutil.rmtree(tmp, ignore_errors=True)

        # fully exit collect mode
        self.collect_running = False
        self._collect_status = "IDLE"
        self.collect_locked_label = None
        self.collect_waiting_for_hand = False
        self._collect_req_name = None
        self.collect_name = None

    def _exit_collect_to_background(self):
        # stop collection mode only (do not quit app)
        self.collect_cancelled = True
        self.collect_running = False
        self.collect_done = False
        self.collect_paused = True
        self._collect_status = "IDLE"

        # reset collector-specific state
        self.collect_locked_label = None
        self.collect_waiting_for_hand = False
        self._collect_req_name = None

        # restore normal runtime modes
        self.suspend_actions = False
        if self._prev_mouse_mode is not None:
            self.mouse_mode = self._prev_mouse_mode
            self._prev_mouse_mode = None
        if self._prev_hand_mode is not None:
            self.hand_mode = self._prev_hand_mode
            self._prev_hand_mode = None

        # "minimize to background" behavior
        self.want_camera_view = False         # hides OpenCV window like your toggle
        if self.gui is not None:
            try:
                self.gui.root.after(0, self.gui.hide)  # hide Tk GUI too
            except Exception:
                pass

        print("[COLLECT] Exit to background")

    

    def _delete_gesture_and_vectors(self, gesture_name: str):
        gesture_name = (gesture_name or "").strip()
        if not gesture_name:
            print("[DELETE] empty name")
            return

        print("[DELETE] Removing gesture:", gesture_name)

        # 1) Remove from GestureList.json
        gesturelist_remove(gesture_name)

        # 2) Remove from dataset
        ok = dataset_delete_label(gesture_name)
        if ok:
            print("[DELETE] Dataset updated for:", gesture_name)
        else:
            print("[DELETE] Dataset not found or no changes needed.")

        # 3) Delete per-gesture vector folder (if exists)
        vec_folder = os.path.join(DATA_DIR, gesture_name)
        if os.path.isdir(vec_folder):
            shutil.rmtree(vec_folder, ignore_errors=True)
            print("[DELETE] Removed folder:", vec_folder)

        # 4) Reload classifier
        try:
            if self.classifier:
                self.classifier.load_dataset()
                print("[DELETE] Classifier reloaded.")
        except Exception as e:
            print("[DELETE] Reload classifier failed:", e)


    # ---------- main loop ----------

    def run(self):
        if not self.enable_camera:
            print("[RUN] Camera disabled; run() will not start.")
            return

        while self.running:

            # ---- apply pending requests (main thread) ----
            self._req_cycle_hand_mode = False
            self._req_cycle_mouse_mode = False

            if self._req_cycle_hand_mode:
                self._req_cycle_hand_mode = False
                self.cycle_hand_mode()

            if self._req_cycle_mouse_mode:
                self._req_cycle_mouse_mode = False
                self.cycle_mouse_mode()

            if self._req_toggle_camera:
                self._req_toggle_camera = False
                self.want_camera_view = not self.want_camera_view
                print("[MAIN] want_camera_view =", self.want_camera_view, flush=True)

            if self._req_toggle_gui:
                self._req_toggle_gui = False
                self.want_gui = not self.want_gui
                print("[MAIN] want_gui =", self.want_gui, flush=True)

                # send to GUI thread
                if self.enable_gui:
                    try:
                        self._gui_queue.put(("SHOW",) if self.want_gui else ("HIDE",))
                    except Exception:
                        pass

            if self._collect_req_name is not None:
                name = self._collect_req_name
                self._collect_req_name = None
                self._start_collect_mode(name)


            ret, frame_raw = self.cap.read()
            if not ret:
                continue

            # Use RAW frame for tracking (MediaPipe + feature extraction)
            frame_for_tracking = frame_raw

            # Use FLIPPED frame only for display (so it looks mirror-like to you)
            frame = cv2.flip(frame_raw, 1)


            # Decide what frame MediaPipe should see
            if self.cam_apply_to_tracking:
                frame_for_mp = self._apply_camera_adjustments(frame_for_tracking.copy())
            else:
                frame_for_mp = frame_for_tracking

            frame_rgb = cv2.cvtColor(frame_for_mp, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((time.perf_counter() - self._mp_start_t) * 1000)

            # Decide what you want to display
            frame_display = self._apply_camera_adjustments(frame.copy())

            
            pointer_hand = None
            action_hand = None

            # track best point gesture for auto
            detected = []
            best_point_conf = 0.0
            best_point_hand = None

            res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if res.hand_landmarks and res.handedness:
                for i in range(min(len(res.hand_landmarks), len(res.handedness))):
                    hand_lm = res.hand_landmarks[i]  # list of 21 landmarks
                    # handedness[i] is a list of Category; take top category
                    raw_label = res.handedness[i][0].category_name
                    label = self.normalize_handedness(raw_label)

                    # compute both normal + mirrored predictions, choose best
                    # --- Base prediction (no mirroring) ---
                    feat = landmarks_to_feature_vector(hand_lm, mirror=False)
                    pred1, conf1 = self.classifier.predict(feat)
                    pred = pred1
                    conf = conf1
                    # --- Mirror-invariant inference ONLY for non-directional gestures ---
                    if pred1 not in DIRECTIONAL_GESTURES:
                        feat_m = landmarks_to_feature_vector(hand_lm, mirror=True)
                        pred2, conf2 = self.classifier.predict(feat_m)

                        if conf2 > conf1:
                            pred = pred2
                            conf = conf2


                    if conf < GESTURE_CONF_THRESHOLD:
                        pred = "none"
                        conf = 0.0
                    # enforce one-hand-only gestures
                    pred = self.enforce_hand_suffix(pred, label)  # label is this hand's "Left"/"Right"
                    detected.append((label, hand_lm, pred, conf))

                    if pred == "point" and conf > best_point_conf:
                        best_point_conf = conf
                        best_point_hand = (label, hand_lm, pred, conf)

            if self.collect_running:
                self._collect_step(detected)


            # ---- draw landmarks (Tasks API) ----
            if self.show_hand_vectors and detected:
                h, w = frame_display.shape[:2]
                for (_, hand_lm, _, _) in detected:
                    for p in hand_lm:
                        cx = int((1.0 - p.x) * w)  # keep this if your display is flipped
                        cy = int(p.y * h)
                        cv2.circle(frame_display, (cx, cy), 2, (255, 0, 0), -1)




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
                tip = hand_lm[8]  # index fingertip (UNFLIPPED coords)

                # Convert to DISPLAY coords (because frame_display is FLIPPED)
                x = 1.0 - tip.x
                y = tip.y

                if self.mouse_mode == "DISABLED":
                    pass
                elif self.mouse_mode == "CURSOR":
                    self._cursor_move(x, y)   # use flipped-x
                else:
                    dx = (x - 0.5) * self._joy_gain
                    dy = (y - 0.5) * self._joy_gain
                    dx = self._apply_deadzone(dx)
                    dy = self._apply_deadzone(dy)
                    self._camera_move(dx, dy)

                # draw red dot on fingertip (on FLIPPED display frame)
                h, w = frame_display.shape[:2]
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame_display, (cx, cy), 8, (0, 0, 255), -1)


            # ---- action execution (MULTI-HAND for multi_keyboard) ----
            # Stop holds for hands that disappeared
            seen_labels = set([lbl for (lbl, _, _, _) in action_hands])

            for lbl in ["Left", "Right"]:
                if lbl not in seen_labels:
                    prev_hold_token = self.prev_hold_action_by_hand.get(lbl)
                    if prev_hold_token is not None:
                        self._hold_up(prev_hold_token)
                    self.prev_hold_action_by_hand[lbl] = None
                    self.prev_gesture_by_hand[lbl] = "none"

            mapped_texts = []
            fired_texts = []

            if not self.suspend_actions:
                for (label, _, pred, conf) in action_hands:
                    current_gesture = pred if conf >= GESTURE_CONF_THRESHOLD else "none"
                    mapped_text, fired_text = self._process_action_for_hand(label, current_gesture)
                    mapped_texts.append(f"{label}: {mapped_text}")
                    fired_texts.append(f"{label}: {fired_text}")
            else:
                for lbl in ["Left", "Right"]:
                    prev_hold = self.prev_hold_action_by_hand.get(lbl)
                    if prev_hold is not None:
                        try:
                            prev_hold.stopHold()
                        except Exception:
                            pass
                    self.prev_hold_action_by_hand[lbl] = None
                    self.prev_gesture_by_hand[lbl] = "none"


            # UI: show both hands in multi_keyboard, else show single current gesture
            # ---- Update detected gesture text on camera window ----
            if self.hand_mode == "multi_keyboard":
                gL = self.prev_gesture_by_hand.get("Left", "none")
                gR = self.prev_gesture_by_hand.get("Right", "none")
                gesture_text = f"L={gL} | R={gR}"
            else:
                gesture_text = "nothing"
                if action_hands:
                    lbl = action_hands[0][0]  # "Left" or "Right"
                    gesture_text = self.prev_gesture_by_hand.get(lbl, "none")
                    

            # normalize to "nothing" when none
            if not gesture_text or gesture_text.lower() in ("none", "nothing", "null"):
                gesture_text = "nothing"
            elif gesture_text.lower() == "none":
                gesture_text = "nothing"

            # Make sure the Qt camera window exists
            self._ensure_qt_camera()

            if self.camera_qt is not None:
                self.camera_qt.set_detected_gesture_text(gesture_text)


            mapped_text = " | ".join(mapped_texts) if mapped_texts else "None"
            fired_text = " | ".join(fired_texts) if fired_texts else "NO"
            '''
            mon_txt = "All Screens"
            if self.monitor:
            '''
            mon_txt = f"{self.screen_w}x{self.screen_h}"

            mode_text = f"Hand Mode: {self.hand_mode} | Mouse Mode: {self.mouse_mode} | Monitor: {mon_txt}"

            # Thread-safe UI update
            if self.enable_gui:
                try:
                    self._gui_queue.put((
                        "STATUS",
                        mode_text,
                        gesture_text,
                        f"Mapped: {mapped_text}",
                        f"Fired: {fired_text}",
                    ))
                except Exception:
                    pass


            # ---- PySide6 camera window update ----
            if self.want_camera_view:
                self._ensure_qt_camera()

                if self.collect_running:
                    self._draw_collect_overlay(frame_display)

                if self.camera_qt is not None:

                    if not self.camera_qt.isVisible():
                        self.camera_qt.show()
                        self.camera_qt.raise_()
                        self.camera_qt.activateWindow()

                    # 1️⃣ Update frame
                    self.camera_qt.set_frame_bgr(frame_display)

                    # 2️⃣ Update detected gesture text
                    self.camera_qt.set_detected_gesture_text(gesture_text)

                try:
                    self.qt_app.processEvents()
                except Exception:
                    pass

            else:
                if self.camera_qt is not None and self.camera_qt.isVisible():
                    self.camera_qt.hide()
                    try:
                        self.qt_app.processEvents()
                    except Exception:
                        pass


            # IMPORTANT: no cv2.waitKey anymore
            k = 255
            if self.camera_qt is not None:
                kk = self.camera_qt.pop_last_key()
                if kk is not None:
                    k = kk


            # ---------- KEY HANDLING (SAFE & EXTENDABLE) ----------
            handled = False

            # ===== 1) Gesture collection mode (highest priority) =====
            if self.collect_running:
                if k == 32:  # SPACE
                    self.collect_paused = not self.collect_paused
                    if not self.collect_paused:
                        self.collect_locked_label = None          # allow relock
                    print("[COLLECT]", "PAUSED" if self.collect_paused else "RESUMED")
                    handled = True


                elif k == ord('q'):
                    self._exit_collect_to_background()
                    handled = True


                elif k == ord('s'):
                    self.collect_mode = "ONE_HAND" if self.collect_mode == "TWO_HAND" else "TWO_HAND"

                    # reset progress
                    self.collect_saved = 0
                    self.collect_saved_phase = {"LEFT": 0, "RIGHT": 0}
                    self.collect_phase = "LEFT"

                    # IMPORTANT: force pause + reset timer so next frame can't instantly save
                    self.collect_paused = True
                    self._collect_last_save_t = time.perf_counter()
                    self.collect_locked_label = None

                    print(f"[COLLECT] Mode switched to {self.collect_mode}. Reset. Press SPACE.")
                    handled = True


                elif k == ord('d'):
                    if self.collect_mode == "ONE_HAND":
                        self.collect_target_one = "RIGHT" if self.collect_target_one == "LEFT" else "LEFT"
                        self.collect_saved = 0

                        # IMPORTANT: force pause + reset timer
                        self.collect_paused = True
                        self._collect_last_save_t = time.perf_counter()
                        self.collect_locked_label = None

                        print(f"[COLLECT] Target switched to {self.collect_target_one}. Reset. Press SPACE.")
                        handled = True

                elif k == ord('n'):
                    # Add a "null" gesture sample: a 42D zero vector (no hand vectors)
                    self._add_null_gesture_sample()
                    self._exit_collect_to_background()

                    print("[COLLECT] Null gesture saved. Exiting collect mode.")
                    handled = True


            if self._delete_req_name is not None:
                g = self._delete_req_name
                self._delete_req_name = None
                self._delete_gesture_and_vectors(g)

        # ---- cleanup ----
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if self.hand_landmarker is not None:
                self.hand_landmarker.close()
        except Exception:
            pass

        if self.enable_gui:
            try:
                self._gui_queue.put(None)  # shutdown signal
            except Exception:
                pass

class BootstrapApp:
    def __init__(self):
        self.running = True
        self._req_toggle_camera = False
        self._req_toggle_gui = False
        self._collect_req_name = None
        self._delete_req_name = None

    def set_active_profile(self, profile_id: str) -> bool:
        print("[BOOT] SET_PROFILE received before real app ready:", profile_id, flush=True)
        return True

if __name__ == "__main__":
    import argparse, traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--port", type=int, default=50555)
    args = parser.parse_args()

    cmd_server = None
    try:
        # 1) Start server first with a bootstrap app to prevent crashes from server refusal
        boot = BootstrapApp()
        cmd_server = CommandServer(boot, port=args.port)
        cmd_server.start()
        print("[MAIN] Server thread launched EARLY.", flush=True)

        # Start the app
        app = GestureControllerApp(enable_gui=True, enable_camera=True)

        #Swap the server to use the real app
        cmd_server.app = app
        print("[MAIN] Server now bound to real app.", flush=True)

        app.run()

    except Exception as e:
        print("\n[FATAL] Crashed during startup/run:", e, flush=True)
        traceback.print_exc()
        input("Press Enter to exit...")





