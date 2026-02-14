from . import resources_rc
import subprocess
import sys
import os
import json
import threading
import socket
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice, QTimer, QEvent, QSize, QObject
from PySide6.QtGui import QPixmap, QIcon, QFont, QKeySequence, QKeyEvent , QShortcut
from PySide6.QtWidgets import ( 
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QStatusBar, QMessageBox, QLabel, QPushButton, 
    QLineEdit, QComboBox, QTabBar, QToolButton, QDialog, QScrollArea, 
    QSizePolicy, QFrame, QTextBrowser, QGraphicsDropShadowEffect, QTabWidget, QTextEdit, QDialogButtonBox, QInputDialog, QSlider, QCheckBox,
    QToolBox
)
from pathlib import Path
from ProfileManager import ProfileManager

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 50555
KEYMAP_SPECIAL = {
    Qt.Key_Space: "space",
    Qt.Key_Escape: "esc",
    Qt.Key_Return: "enter",
    Qt.Key_Enter: "enter",
    Qt.Key_Backspace: "backspace",
    Qt.Key_Tab: "tab",
    Qt.Key_Delete: "delete",
    Qt.Key_Insert: "insert",
    Qt.Key_Home: "home",
    Qt.Key_End: "end",
    Qt.Key_PageUp: "pageup",
    Qt.Key_PageDown: "pagedown",
    Qt.Key_Left: "left",
    Qt.Key_Right: "right",
    Qt.Key_Up: "up",
    Qt.Key_Down: "down",
    Qt.Key_Shift: "shift",
    Qt.Key_Control: "ctrl",
    Qt.Key_Alt: "alt",
    Qt.Key_Meta: "meta",
}
KEYSTR_TO_QTKEY = {v: k for k, v in KEYMAP_SPECIAL.items()}
def _profile_manager_path() -> str:
    # profileManager.json is beside ProfileManager.py
    base_dir = os.path.dirname(os.path.abspath(__import__("ProfileManager").__file__))
    return os.path.join(base_dir, "profileManager.json")


DEFAULT_SHORTCUTS = {
    "cycle_hand_mode": "f1",
    "cycle_mouse_mode": "f2",
}


def load_profile_manager_json() -> dict:
    path = _profile_manager_path()
    if not os.path.exists(path):
        return {"profileNames": ["Default"], "shortcuts": dict(DEFAULT_SHORTCUTS)}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    if "profileNames" not in data or not isinstance(data["profileNames"], list):
        data["profileNames"] = ["Default"]

    sc = data.get("shortcuts")
    if not isinstance(sc, dict):
        sc = {}

    merged = dict(DEFAULT_SHORTCUTS)
    for k, v in sc.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            merged[k] = v.strip().lower()

    data["shortcuts"] = merged
    return data


def save_profile_manager_json(data: dict) -> bool:
    path = _profile_manager_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print("[UI] Failed saving profileManager.json:", e)
        return False

def get_pm_setting(name: str, default=None):
    pm = load_profile_manager_json()
    settings = pm.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    return settings.get(name, default)

def set_pm_setting(name: str, value):
    pm = load_profile_manager_json()
    settings = pm.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    settings[name] = value
    pm["settings"] = settings
    save_profile_manager_json(pm)


def set_shortcut_in_profile_manager(action_name: str, key_str: str | None) -> bool:
    data = load_profile_manager_json()

    if "shortcuts" not in data or not isinstance(data["shortcuts"], dict):
        data["shortcuts"] = dict(DEFAULT_SHORTCUTS)

    if not key_str or not str(key_str).strip():
        # Treat NULL/empty as disabled (no binding)
        data["shortcuts"][action_name] = ""
    else:
        data["shortcuts"][action_name] = str(key_str).strip().lower()

    return save_profile_manager_json(data)


def backend_is_running(host=BACKEND_HOST, port=BACKEND_PORT, timeout=1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b"PING")
            resp = s.recv(1024).decode("utf-8", errors="ignore").strip()
            return resp == "PONG"
    except Exception:
        return False


def start_backend_if_needed(proto_path: str, project_root: str, port: int = BACKEND_PORT):
    if backend_is_running(port=port):
        return None  # already running

    return subprocess.Popen(
        [sys.executable, proto_path, "--background", "--port", str(port)],
        cwd=project_root
    )

def keyevent_to_string(event) -> str | None:
    k = event.key()

    # map well-known special keys
    if k in KEYMAP_SPECIAL:
        return KEYMAP_SPECIAL[k]

    # function keys
    if Qt.Key_F1 <= k <= Qt.Key_F35:
        return f"f{k - Qt.Key_F1 + 1}"

    # printable text (letters, digits, symbols)
    txt = (event.text() or "")
    if txt.strip():
        return txt.lower()

    # fallback
    return f"key_{int(k)}"

class AppShortcutFilter(QObject):
    """
    Application-wide shortcut handler.
    Works across ALL windows/dialogs in your app immediately.
    """
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.shortcuts = {}   # action_name -> key_str (e.g. "h", "space", "esc")
        self.callbacks = {}   # action_name -> callable

    def set_shortcuts(self, shortcuts: dict):
        # normalize
        out = {}
        for k, v in (shortcuts or {}).items():
            if isinstance(k, str):
                out[k] = (v or "").strip().lower()
        self.shortcuts = out

    def set_callbacks(self, callbacks: dict):
        self.callbacks = dict(callbacks or {})

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False

        # If a modal key-capture dialog is open, do NOT trigger shortcuts
        modal = QApplication.activeModalWidget()
        if isinstance(modal, KeyCaptureDialog):
            return False

        key_str = keyevent_to_string(event)  # uses your existing function
        if not key_str:
            return False

        key_str = key_str.strip().lower()

        # find matching action
        for action_name, bound in self.shortcuts.items():
            if bound and bound == key_str:
                cb = self.callbacks.get(action_name)
                if cb:
                    cb()
                    return True  # consume so it doesn't type into a textbox etc.

        return False
    
class KeyCaptureDialog(QDialog):
    """
    Modal dialog:
    - Press ANY key (including ESC) → captures key
    - Click 'Set NULL' → explicit null binding
    - Click 'Cancel' → abort
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Key")
        self.setModal(True)
        self.setFixedSize(500, 250)
        self.setStyleSheet("background: #3c384d")
        
        self.captured_key = None  # str | "" | None

        layout = QVBoxLayout(self)

        label = QLabel(
            "Press a key to bind\n\n"
            "• Supports ESC, Shift, Ctrl, arrows, etc.\n"
            "• Use buttons below to cancel or set NULL",
            self
        )
        label.setAlignment(Qt.AlignCenter)
        font = label.font()
        font.setPointSize(14)
        label.setFont(font)
        layout.addWidget(label)

        # --- Buttons ---
        btn_row = QHBoxLayout()

        self.null_btn = QPushButton("Set NULL")
        self.null_btn.setToolTip("Remove key binding (null)")
        self.null_btn.clicked.connect(self._set_null)
        self.null_btn.setStyleSheet("border: 2px solid #e0dde5; border-radius: 8px; font-size: 14px;")
        self.null_btn.setFixedSize(100,30)
        self.null_btn.setFocusPolicy(Qt.NoFocus)
        self.null_btn.setDefault(False)
        self.null_btn.setAutoDefault(False)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("border: 2px solid #e0dde5; border-radius: 8px; font-size: 14px;")
        self.cancel_btn.setFixedSize(100,30)
        self.cancel_btn.setFocusPolicy(Qt.NoFocus)
        self.cancel_btn.setDefault(False)
        self.cancel_btn.setAutoDefault(False)

        btn_row.addStretch(1)
        btn_row.addWidget(self.null_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch(1)
        btn_row.setSpacing(20)

        layout.addLayout(btn_row)

    def _set_null(self):
        self.captured_key = ""
        self.accept()

    def keyPressEvent(self, event):
        s = keyevent_to_string(event)
        if s:
            self.captured_key = s
            self.accept()
            return

        event.ignore()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.ui")
        file = QFile(ui_path)
        if not file.open(QIODevice.ReadOnly):
            raise RuntimeError(f"Unable to open UI file: {ui_path}")
        loader = QUiLoader()
        self.window = loader.load(file, self)
        # --- App-wide shortcut filter (works across all dialogs/windows) ---
        self._shortcut_filter = AppShortcutFilter(self)
        QApplication.instance().installEventFilter(self._shortcut_filter)

        # load + apply now
        self._refresh_shortcuts_runtime()
        
        file.close()
        self._shortcuts_suspended = False  # used when capturing a key
        self._shortcut_map = load_profile_manager_json().get("shortcuts", dict(DEFAULT_SHORTCUTS))

        # install a global event filter so shortcuts work across ALL dialogs/windows
        QApplication.instance().installEventFilter(self)

        if self.window is None:
            raise RuntimeError(loader.errorString())
        try:
            self.profiles = ProfileManager.readFile("profileManager.json")
        except Exception as e:
            print("[UI] Failed to read profileManager.json, using empty ProfileManager:", e)
            self.profiles = ProfileManager()

        #self.tabs refer to the entire tab widget not just the tab bar
        self.tabs = self.window.findChild(QTabWidget, "tabWidget")
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.setTabText(0, "Default")
        # the new tab button is actually a tab itself, i just make it so that it behaves like a new tab button
        self.tabs.addTab(QWidget(), "")
        self.last_tab_index = self.tabs.count() - 1
        self.tabs.tabBar().setTabButton(self.last_tab_index, QTabBar.RightSide, None)
        self.tabs.tabBar().setTabData(self.last_tab_index, "add_tab_button")
        self.tabs.tabBar().setTabText(self.last_tab_index, "")
        self.tabs.tabBar().tabBarClicked.connect(self.new_tab_button)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        #spacer is a croner widget for the tab bar, can only implement a corner widget through python, the purpose of this spacer is to prevent
        # the tab bar overlapping with the main power button
        spacer = QPushButton("                        ")
        spacer.setFlat(True)
        spacer.setFocusPolicy(Qt.NoFocus)
        spacer.setStyleSheet ("""
        QPushButton {
            background: transparent;
            border: none;
            outline: none;
        }

        QPushButton:hover {
            background: transparent;
        }

        QPushButton:pressed {
            background: transparent;
        }

        QPushButton:focus {
            outline: none;
        }
        """)

        self.tabs.setCornerWidget(spacer, Qt.TopRightCorner)
        # ---- Power button ----
        self.power_button = self.window.findChild(QPushButton, "PowerButton")
        assert self.power_button is not None, "PowerButton not found (check objectName in .ui)"
        self.power_button.clicked.connect(self.main_power_button)

        # ---- Camera button ----
        self.camera_button = self.window.findChild(QPushButton, "camera_button")
        assert self.camera_button is not None, "camera_button not found (check objectName in .ui)"
        self.camera_button.clicked.connect(self.on_camera_clicked)
        
        # ---- Setting button ----
        self.setting_button = self.window.findChild(QPushButton, "setting_button")
        assert self.setting_button is not None, "setting_button not found (check objectName in .ui)"
        self.setting_button.clicked.connect(self.setting_dialog)
        
        # ---- Library button ----
        self.library_button = self.window.findChild(QPushButton, "library_button")
        assert self.setting_button is not None, "library_button not found (check objectName in .ui)"
        self.library_button.clicked.connect(self.on_library_clicked)
        
        # got rid of close button for the default/first tab
        default_close_button = self.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        default_close_button.hide()

        self.scroll_container = self.window.findChild(QScrollArea, "scrollArea")
        self.scroll_content = self.scroll_container.widget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_container.setWidgetResizable(True)
        
        self.new_gesture_button = self.window.findChild(QPushButton, "addition_button")
        self.new_gesture_button.clicked.connect(self.new_gesture_button_function)
        self.tabs.tabBarDoubleClicked.connect(self.tab_rename)
        QTimer.singleShot(800, lambda: self._on_tab_changed(self.tabs.currentIndex()))
        # ---- shortcut system (single-key safe) ----
        self._shortcut_map = {}
        self._reload_shortcut_map()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)


        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        proto_path = os.path.join(project_root, "prototypeV2.py")

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Start ONE background process (only once)
        self._backend_proc = start_backend_if_needed(
            proto_path=proto_path,
            project_root=project_root,
            port=50555
        )
        # Force backend to load active profile on startup (wait until backend answers PING)
        def _boot_set_profile():
            if backend_is_running():
                self._on_tab_changed(self.tabs.currentIndex())
            else:
                # try again shortly (backend still booting)
                QTimer.singleShot(300, _boot_set_profile)

        QTimer.singleShot(300, _boot_set_profile)

        v = get_pm_setting("sensitivity", 50)
        self.send_cmd(f"SET_SENS {int(v)}")

        
        self.BASE_DIR = Path(__file__).parent
        self.PARENT_DIR = self.BASE_DIR.parent
        # ---- profile json watcher ----
        self._profiles_snapshot = {}  # filepath -> mtime
        self._profiles_timer = QTimer(self)
        self._profiles_timer.timeout.connect(self._tick_profiles_refresh)
        self._profiles_timer.start(700)

        RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # ---- Load existing profiles into tabs (DO NOT use new_tab_button here) ----
        json_files = sorted(self.PARENT_DIR.glob("profile_*.json"))
        # ===== Load Default.json into EXISTING tab 0 =====
        default_idx = 0
        default_profile_id = "Default"

        tab = self.tabs.widget(default_idx)
        scroll = tab.findChild(QScrollArea)

        if scroll and scroll.property("individual_sub_bar_container") is True:
            layout = scroll.widget().layout()

            path = self._profile_path(default_profile_id)
            if not os.path.exists(path):
                print(f"[UI] Default profile missing -> NOT clearing UI: {path}")
            else:
                data = self._load_profile_json(default_profile_id)

                actions = data.get("Actions", None)
                if not isinstance(actions, list):
                    actions = []

                print(f"[UI] Loaded Default from {path}, actions={len(actions)}")

                # ONLY clear if we successfully loaded Default.json
                self._clear_layout(layout)

                class _ActShim:
                    def __init__(self, d): self._d = d
                    def getGName(self): return self._d.get("G_name")
                    def getKeyPressed(self): return self._d.get("key_pressed")
                    def getInputType(self): return self._d.get("input_type")
                    def getName(self): return self._d.get("name")

                for row in actions:
                    if isinstance(row, dict):
                        self.build_action_row(
                            layout,
                            profile_id=default_profile_id,
                            act=_ActShim(row)
                        )
            
                # ===== Load other profile_*.json into NEW tabs =====
                for f in json_files:
                    profile_id = f.stem.replace("profile_", "", 1)

                    # skip if it would be Default (just in case you ever have profile_Default.json)
                    if profile_id.lower() == "default":
                        continue

                    tab_index = self._add_profile_tab(profile_id)

                    # populate actions into that tab
                    tab = self.tabs.widget(tab_index)
                    scroll = tab.findChild(QScrollArea)
                    if scroll and scroll.property("individual_sub_bar_container") is True:
                        layout = scroll.widget().layout()
                        current_profile = self.profiles.loadProfile(profile_id)
                        if current_profile is not None:
                            for act in (current_profile.getActionList() or []):
                                self.build_action_row(layout, profile_id=profile_id, act=act)

        # --- GestureList.json watcher (same polling refresh logic as library dialog) ---
        self._gesturelist_refreshing = False

        try:
            p = self._gesturelist_path()
            self._gesturelist_last_mtime = os.path.getmtime(p) if os.path.exists(p) else None
        except Exception:
            self._gesturelist_last_mtime = None

        self._profiles_snapshot = {}  # filepath -> mtime
        self._profiles_timer = QTimer(self)
        self._profiles_timer.timeout.connect(self._tick_profiles_refresh)
        self._profiles_timer.start(700)  # 0.7s
        
        self._gesturelist_timer = QTimer(self)
        self._gesturelist_timer.timeout.connect(self._tick_gesturelist_refresh)
        self._gesturelist_timer.start(500)

        # ensure we are actually on Default tab (if it exists)
        for i in range(self.tabs.count()):
            if self.tabs.tabBar().tabData(i) == "add_tab_button":
                continue
            if self.tabs.tabText(i).strip() == "Default":
                self.tabs.setCurrentIndex(i)
                break

        QTimer.singleShot(50, self._reload_active_tab_actions)

    def _sync_profile_tabs_from_disk(self):
        """
        Make tabs match the profile JSON files on disk.
        Uses your existing tab add/remove handlers.
        """
        # Desired profile IDs based on files
        desired = {"Default"}

        for f in self.PARENT_DIR.glob("profile_*.json"):
            profile_id = f.stem.replace("profile_", "", 1)
            if profile_id and profile_id.lower() != "default":
                desired.add(profile_id)

        # Existing profile IDs from tabs
        existing = []
        for i in range(self.tabs.count()):
            if self.tabs.tabBar().tabData(i) == "add_tab_button":
                continue
            pid = self._profile_id_for_tab(i)
            if pid:
                existing.append(pid)

        existing_set = set(existing)

        # Add missing tabs
        for pid in sorted(desired - existing_set):
            if pid == "Default":
                continue  # Default is tab 0 already
            self._add_profile_tab(pid)

        # Remove tabs whose files are gone (skip Default)
        for i in reversed(range(self.tabs.count())):
            if self.tabs.tabBar().tabData(i) == "add_tab_button":
                continue
            pid = self._profile_id_for_tab(i)
            if not pid or pid == "Default":
                continue
            if pid not in desired:
                # This uses your real delete logic (removes tab + deletes JSON if needed)
                # If you DON'T want it to delete files during sync, tell me and I’ll adjust.
                self.close_tab(i)

    def _tick_profiles_refresh(self):
        # Files to watch (based on YOUR current path logic)
        watch_files = []

        # profileManager.json (your loader uses _profile_manager_path())
        try:
            pm_path = _profile_manager_path()
            watch_files.append(pm_path)
        except Exception:
            pm_path = None

        # Profiles folder (your profiles live in self.PARENT_DIR)
        base = self.PARENT_DIR
        watch_files.append(str(base / "Default.json"))
        watch_files.extend([str(p) for p in base.glob("profile_*.json")])

        # Build current snapshot
        current = {}
        changed = False

        for p in watch_files:
            try:
                mt = os.path.getmtime(p) if os.path.exists(p) else None
            except Exception:
                mt = None
            current[p] = mt
            if self._profiles_snapshot.get(p) != mt:
                changed = True

        # Detect deleted profile files that were previously present
        for oldp in list(self._profiles_snapshot.keys()):
            if oldp not in current:
                changed = True

        if not changed:
            return

        self._profiles_snapshot = current
        print("[UI] Detected profile JSON change -> refreshing tabs/actions/shortcuts")

        # 1) Shortcuts (Settings window also benefits)
        try:
            self._install_shortcuts()
        except Exception as e:
            print("[UI] _install_shortcuts failed:", e)

        # 2) Sync tabs with disk
        try:
            self._sync_profile_tabs_from_disk()
        except Exception as e:
            print("[UI] _sync_profile_tabs_from_disk failed:", e)

        # 3) Reload current tab action list + tell backend to reload mapping
        try:
            self._reload_active_tab_actions()
        except Exception as e:
            print("[UI] _reload_active_tab_actions failed:", e)


    def _refresh_shortcuts_runtime(self):
        pm = load_profile_manager_json()
        sc = pm.get("shortcuts", DEFAULT_SHORTCUTS)
        if not isinstance(sc, dict):
            sc = dict(DEFAULT_SHORTCUTS)

        # callbacks (what each shortcut actually does)
        callbacks = {
            "cycle_hand_mode": lambda: print("cycle_hand_mode ->", self.send_cmd("CYCLE_HAND_MODE")),
            "cycle_mouse_mode": lambda: print("cycle_mouse_mode ->", self.send_cmd("CYCLE_MOUSE_MODE")),
            "toggle_camera":    lambda: print("toggle_camera ->", self.send_cmd("TOGGLE_CAMERA")),
            "toggle_vectors":   lambda: print("toggle_vectors ->", self.send_cmd("TOGGLE_VECTORS")),
            "reload_profile":   lambda: print("reload_profile ->", self.send_cmd("RELOAD_PROFILE")),
            "quit":             lambda: self.main_power_button(),
        }

        self._shortcut_filter.set_shortcuts(sc)
        self._shortcut_filter.set_callbacks(callbacks)

    def eventFilter(self, obj, event):
        if self._shortcuts_suspended:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.KeyPress:
            # convert Qt keypress to your string format ("h", "space", "esc", "left", etc.)
            s = keyevent_to_string(event)
            if not s:
                return super().eventFilter(obj, event)

            # reload latest shortcuts (cheap enough, but you can cache)
            shortcuts = load_profile_manager_json().get("shortcuts", dict(DEFAULT_SHORTCUTS))

            # match + trigger
            if s == (shortcuts.get("cycle_hand_mode") or "").strip().lower():
                print("[UI] cycle hand mode ->", self.send_cmd("CYCLE_HAND_MODE"))
                return True

            if s == (shortcuts.get("cycle_mouse_mode") or "").strip().lower():
                print("[UI] cycle mouse mode ->", self.send_cmd("CYCLE_MOUSE_MODE"))
                return True

            if s == (shortcuts.get("toggle_camera") or "").strip().lower():
                self.on_camera_clicked()
                return True

            if s == (shortcuts.get("quit") or "").strip().lower():
                self.main_power_button()
                return True

        return super().eventFilter(obj, event)

    def _profile_id_for_tab(self, idx: int) -> str:
        if idx == 0:
            return "Default"
        return self.tabs.tabText(idx).strip()
    
    def _keystr_to_qkeysequence(self, s: str) -> QKeySequence | None:
        if not s:
            return None
        s = s.strip().lower()

        # Named specials using your existing mapping
        qtkey = KEYSTR_TO_QTKEY.get(s)
        if qtkey is not None:
            return QKeySequence(qtkey)

        # Function keys: f1..f35
        if s.startswith("f") and s[1:].isdigit():
            n = int(s[1:])
            if 1 <= n <= 35:
                return QKeySequence(getattr(Qt, f"Key_F{n}"))

        # Single character (letters/digits/symbols)
        if len(s) == 1:
            return QKeySequence(s)

        # Fallback: let Qt parse (supports combos if you add them later)
        seq = QKeySequence(s)
        return None if seq.isEmpty() else seq

    def _install_shortcuts(self):
        """
        Creates/refreshes QShortcut objects for the UI.
        Uses ApplicationShortcut so it works even if a child widget has focus.
        """
        # remove old ones
        if hasattr(self, "_qt_shortcuts"):
            for sc in getattr(self, "_qt_shortcuts", []):
                try:
                    sc.setEnabled(False)
                    sc.deleteLater()
                except Exception:
                    pass
        self._qt_shortcuts = []

        pm = load_profile_manager_json()
        shortcuts = pm.get("shortcuts", {})
        if not isinstance(shortcuts, dict):
            shortcuts = {}

        # apply defaults if missing
        defaults = dict(DEFAULT_SHORTCUTS)
        changed = False
        for k, v in defaults.items():
            if k not in shortcuts:
                shortcuts[k] = v
                changed = True
        if changed:
            pm["shortcuts"] = shortcuts
            save_profile_manager_json(pm)

        def bind(action_name: str, callback):
            key_str = (shortcuts.get(action_name) or "").strip().lower()
            seq = self._keystr_to_qkeysequence(key_str)
            if seq is None:
                return  # no shortcut set
            sc = QShortcut(seq, self.window)  # IMPORTANT: attach to the loaded UI root
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(callback)
            self._qt_shortcuts.append(sc)

        # Bind shortcuts to your existing actions
        bind("cycle_hand_mode", lambda: self.send_cmd("CYCLE_HAND_MODE"))
        bind("cycle_mouse_mode", lambda: self.send_cmd("CYCLE_MOUSE_MODE"))
        bind("toggle_camera", lambda: self.on_camera_clicked())
        bind("toggle_vectors", lambda: self.send_cmd("TOGGLE_VECTORS"))
        bind("reload_profile", lambda: self.send_cmd("RELOAD_PROFILE"))
        bind("quit", lambda: self.main_power_button())

    def _reload_shortcut_map(self):
        """
        Loads shortcuts from profileManager.json into a fast lookup map.
        Uses your existing file format:
        { "profileNames": [...], "shortcuts": { "cycle_hand_mode": "f1", ... } }
        """
        pm = load_profile_manager_json()
        sc = pm.get("shortcuts", {})
        if not isinstance(sc, dict):
            sc = {}

        # normalize to lowercase strings
        self._shortcut_map = {k: (v or "").strip().lower() for k, v in sc.items()}

    def _dispatch_shortcut(self, key_str: str) -> bool:
        """
        Returns True if key_str matched and we handled it.
        """
        sc = self._shortcut_map

        if key_str and key_str == sc.get("cycle_hand_mode", ""):
            print("[UI] shortcut: cycle_hand_mode")
            print(self.send_cmd("CYCLE_HAND_MODE"))
            return True

        if key_str and key_str == sc.get("cycle_mouse_mode", ""):
            print("[UI] shortcut: cycle_mouse_mode")
            print(self.send_cmd("CYCLE_MOUSE_MODE"))
            return True

        if key_str and key_str == sc.get("toggle_camera", ""):
            print("[UI] shortcut: toggle_camera")
            self.on_camera_clicked()
            return True

        if key_str and key_str == sc.get("quit", ""):
            print("[UI] shortcut: quit")
            self.main_power_button()
            return True

        return False
    def _sync_mode_dropdowns(self, hand_mode_options: QComboBox, mouse_mode_options: QComboBox):
        resp = self.send_cmd("GET_MODES")
        # expecting: "OK <hand> <mouse>"
        parts = resp.split()
        if len(parts) < 3 or parts[0] != "OK":
            print("[UI] GET_MODES failed:", resp)
            return

        hand = parts[1].strip().lower()
        mouse = parts[2].strip().upper()

        backend_to_ui_hand = {
            "right": "Right pointer",
            "left": "Left pointer",
            "auto": "Auto",
            "multi_keyboard": "MultiKB",
            "multikb": "MultiKB",  # if you ever used this name
        }
        backend_to_ui_mouse = {
            "DISABLED": "Disabled",
            "CAMERA": "Camera",
            "CURSOR": "Cursor",
        }

        # Block signals so setting current text doesn't trigger _apply_* immediately
        hb = hand_mode_options.blockSignals(True)
        mb = mouse_mode_options.blockSignals(True)

        hand_mode_options.setCurrentText(backend_to_ui_hand.get(hand, "MultiKB"))
        mouse_mode_options.setCurrentText(backend_to_ui_mouse.get(mouse, "Disabled"))

        hand_mode_options.blockSignals(hb)
        mouse_mode_options.blockSignals(mb)

    def eventFilter(self, obj, event):
        # Don’t steal keys while KeyCaptureDialog is open
        mw = QApplication.activeModalWidget()
        if isinstance(mw, KeyCaptureDialog):
            return super().eventFilter(obj, event)

        if event.type() == QEvent.KeyPress:
            # Convert Qt event to your string form ("h", "space", "esc", "left", etc.)
            key_str = keyevent_to_string(event)

            if key_str:
                handled = self._dispatch_shortcut(key_str)
                if handled:
                    event.accept()
                    return True  # stop propagation so it doesn't type into textboxes

        return super().eventFilter(obj, event)
    
    def user_manual_dialog(self):
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        usage_instructions_path = os.path.join(BASE_DIR, "user_manual_image", "image14.png")
        create_profile_path = os.path.join(BASE_DIR, "user_manual_image", "image8.png")
        delete_profile_path = os.path.join(BASE_DIR, "user_manual_image", "image7.png")
        rename_profile_path_1 = os.path.join(BASE_DIR, "user_manual_image", "image35.png")
        rename_profile_path_2 = os.path.join(BASE_DIR, "user_manual_image", "image44.png")
        swap_profile_path = os.path.join(BASE_DIR, "user_manual_image", "image17.png")
        create_gesture_path_1 = os.path.join(BASE_DIR, "user_manual_image", "image18.png")
        create_gesture_path_2 = os.path.join(BASE_DIR, "user_manual_image", "image16.png")
        
        def make_page(icon_path_1, text_1, icon_path_2, text_2):
            
            page = QWidget()
            layout = QVBoxLayout(page)

            # Image
            icon = QLabel()
            pix = QPixmap(icon_path_1)
            icon.setPixmap(
                pix.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            icon.setStyleSheet("""
                QLabel {
                    border: 2px solid white;
                }
            """)


            # Text
            label = QLabel(text_1)
            f = label.font()
            f.setPointSize(16)
            label.setFont(f)
            label.setWordWrap(True)

            layout.addWidget(label)
            layout.addWidget(icon)
            
            if (icon_path_2 is not None and text_2 is not None):
                icon2 = QLabel()
                pix = QPixmap(icon_path_1)
                icon2.setPixmap(
                    pix.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                icon2.setStyleSheet("""
                    QLabel {
                        border: 2px solid white;
                    }
                """)

                label2 = QLabel(text_2)
                f = label2.font()
                f.setPointSize(16)
                label2.setFont(f)
                label2.setWordWrap(True)
                
                layout.addWidget(label2)
                layout.addWidget(icon2)
                
            layout.addStretch()

            return page
        
        dialog = QDialog(self)
        dialog.setWindowTitle("User Manual")
        dialog.setFixedSize(600, 800)
        dialog.setModal(True)
        
        top_frame = QFrame(dialog)
        top_frame.setGeometry(0, 0, 600, 80)
        top_frame.setStyleSheet("background-color: #030013;")
        
        label_title = QLabel("User Manual", top_frame)
        label_title.setGeometry(30, 20, 220, 40)
        label_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = label_title.font()
        f.setPointSize(20)
        label_title.setFont(f)
        
        dialog_scroll = QScrollArea(dialog)
        dialog_scroll.setGeometry(0, 80, 600, 720)
        dialog_scroll.setStyleSheet("background: #3c384d; border: none;")
        dialog_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        dialog_scroll_content = QWidget()
        dialog_scroll.setWidget(dialog_scroll_content)
        dialog_scroll_layout = QVBoxLayout(dialog_scroll_content)
        dialog_scroll_layout.setAlignment(Qt.AlignTop)
        dialog_scroll_layout.setContentsMargins(20, 30, 20, 10)
        dialog_scroll_layout.setSpacing(20)
        dialog_scroll.setWidgetResizable(True)
        
        toolbox = QToolBox()
        toolbox.setStyleSheet("""
                                QToolBox::tab {
                                    font-size: 20px;
                                    font-weight: bold;
                                    background: #030013;
                                    border-radius: 8px;
                                }
                                """)

        toolbox.addItem(make_page(usage_instructions_path, "This is what you will see when the application is opened.", None, None), "Usage Instructions:")
        
        create_profile = "Create profiles to save a preset of gesture controls.\n\n1. Click on the + button near the profile tabs at the top of the application to create a new profile.\n\n2. The + button will always be to the right of the profile tabs."
        toolbox.addItem(make_page(create_profile_path, create_profile, None, None), "Create Profile:")
        
        delete_profile = "Delete custom profiles.\n\n1. Click on the x button at the right of the profile tab to delete the profile.\n\n2. The default profile cannot be deleted."
        toolbox.addItem(make_page(delete_profile_path, delete_profile, None, None), "Delete Profile:")
        
        rename_profile = "Rename their profile to another name.\n\n1. Double click on the name of the profile in the profile tab to open a window that allows you to enter the new profile name"
        rename_profile_2 = "\n2. Hit OK once you enter the name you want.\n\n3. The default profile cannot be renamed."
        
        toolbox.addItem(make_page(rename_profile_path_1, rename_profile, rename_profile_path_2, rename_profile_2), "Rename Profile:")
        
        swap_profile = "Swap between available profiles.\n\n1. Click on a profile in the profile tab to swap to the profile you want to use."
        toolbox.addItem(make_page(swap_profile_path, swap_profile, None, None), "Swap Profiles:")
        
        create_gesture_1 = "Create a gesture in a custom profile.\n\n1. Click on the + button at the top right of the profile below the power button to create a new gesture."
        create_gesture_2 = "\n2. A tab like this will be created in the profile after pressing the button."
        toolbox.addItem(make_page(create_gesture_path_1, create_gesture_1, create_gesture_path_2, create_gesture_2), "Create Gesture:")
        
        dialog_scroll_layout.addWidget(toolbox)
        dialog.exec()


    def setting_dialog(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        instruction_path = os.path.join(BASE_DIR, "resource", "Chat-Bubble-Square-Question--Streamline-Core.png")
        
        def on_shortcut_button_clicked(button: QPushButton, shortcut_name: str):
            dlg = KeyCaptureDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return

            val = dlg.captured_key
            if val is None:
                return

            if val == "":
                button.setText("NULL")
                set_shortcut_in_profile_manager(shortcut_name, "")
            else:
                button.setText(val)
                set_shortcut_in_profile_manager(shortcut_name, val)
            self._install_shortcuts()
            self._refresh_shortcuts_runtime()

        
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setFixedSize(500, 800)
        dialog.setModal(True)
        
        top_frame = QFrame(dialog)
        top_frame.setGeometry(0, 0, 500, 80)
        top_frame.setStyleSheet("background-color: #030013;")
        
        label_title = QLabel("Settings", top_frame)
        label_title.setGeometry(30, 20, 220, 40)
        label_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = label_title.font()
        f.setPointSize(20)
        label_title.setFont(f)
        
        instruction_button = QPushButton(top_frame)
        instruction_button.setGeometry(400, 10, 60, 60)
        instruction_button.setIcon(QIcon(instruction_path))
        instruction_button.setIconSize(QSize(50,50))
        instruction_button.setStyleSheet("""
                                    QPushButton {background: transparent; border: none; border-radius: 8px;}
                                    QPushButton:hover {background: #252438;}
                                    """)
        instruction_button.clicked.connect(self.user_manual_dialog)
        
        dialog_scroll = QScrollArea(dialog)
        dialog_scroll.setGeometry(0, 80, 500, 720)
        dialog_scroll.setStyleSheet("background: #3c384d; border: none;")
        dialog_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        dialog_scroll_content = QWidget()
        dialog_scroll.setWidget(dialog_scroll_content)
        dialog_scroll_layout = QVBoxLayout(dialog_scroll_content)
        dialog_scroll_layout.setAlignment(Qt.AlignTop)
        dialog_scroll_layout.setContentsMargins(20, 30, 20, 10)
        dialog_scroll_layout.setSpacing(20)
        dialog_scroll.setWidgetResizable(True)
        
        hand_mode_setting = QWidget()
        hand_mode_setting.setFixedHeight(80)
        hand_mode_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(hand_mode_setting)
        
        hand_mode_title = QLabel("Hand mode:", hand_mode_setting)
        hand_mode_title.setGeometry(20,20,230, 40)
        hand_mode_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = hand_mode_title.font()
        f.setPointSize(16)
        hand_mode_title.setFont(f)
        
        hand_mode_options = QComboBox(hand_mode_setting)
        hand_mode_options.setGeometry(320,20,110,40)
        hand_mode_options.setStyleSheet("""color: #030013; background: #e0dde5; border-radius: 8px;
                                            
                                        """)
        
        hand_mode_options.addItems([
            "Right pointer",
            "Left pointer",
            "Auto",
            "MultiKB"
        ])

        for i in range(hand_mode_options.count()):
            hand_mode_options.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            
        
        hand_mode_cycle_setting = QWidget()
        hand_mode_cycle_setting.setFixedHeight(80)
        hand_mode_cycle_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(hand_mode_cycle_setting)
        
        hand_mode_cycle_label = QLabel("Hand mode cycle key:", hand_mode_cycle_setting)
        hand_mode_cycle_label.setGeometry(20,20,230, 40)
        hand_mode_cycle_label.setStyleSheet("color: #e0dde5; background: transparent;")
        f = hand_mode_cycle_label.font()
        f.setPointSize(16)
        hand_mode_cycle_label.setFont(f)
        
        hand_mode_cycle_input = QPushButton(hand_mode_cycle_setting)
        hand_mode_cycle_input.setGeometry(320,25,110,30)
        hand_mode_cycle_input.setStyleSheet("color: #030013; background: #e0dde5; font-size: 12px;")  
        
        mouse_mode_setting = QWidget()
        mouse_mode_setting.setFixedHeight(80)
        mouse_mode_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(mouse_mode_setting)
        
        mouse_mode_title = QLabel("Mouse mode:", mouse_mode_setting)
        mouse_mode_title.setGeometry(20,20,230, 40)
        mouse_mode_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = mouse_mode_title.font()
        f.setPointSize(16)
        mouse_mode_title.setFont(f)
        
        mouse_mode_options = QComboBox(mouse_mode_setting)
        mouse_mode_options.setGeometry(320,20,110,40)
        mouse_mode_options.setStyleSheet("""color: #030013; background: #e0dde5; border-radius: 8px; 
                                            
                                        """)
        
        mouse_mode_options.addItems([
            "Disabled",
            "Camera",
            "Cursor",
        ])
                
        for i in range(mouse_mode_options.count()):
            mouse_mode_options.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            
        mouse_mode_cycle_setting = QWidget()
        mouse_mode_cycle_setting.setFixedHeight(80)
        mouse_mode_cycle_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(mouse_mode_cycle_setting)
        
        mouse_mode_cycle_label = QLabel("Mouse mode cycle key:", mouse_mode_cycle_setting)
        mouse_mode_cycle_label.setGeometry(20,20,230, 40)
        mouse_mode_cycle_label.setStyleSheet("color: #e0dde5; background: transparent;")
        f = mouse_mode_cycle_label.font()
        f.setPointSize(16)
        mouse_mode_cycle_label.setFont(f)
        
        mouse_mode_cycle_input = QPushButton(mouse_mode_cycle_setting)
        mouse_mode_cycle_input.setGeometry(320,25,110,30)
        mouse_mode_cycle_input.setStyleSheet("color: #030013; background: #e0dde5; font-size: 12px;")
            
        # set initial text from profileManager.json
        pm = load_profile_manager_json()

        hand_mode_cycle_input.clicked.connect(lambda: on_shortcut_button_clicked(hand_mode_cycle_input, "cycle_hand_mode"))
        mouse_mode_cycle_input.clicked.connect(lambda: on_shortcut_button_clicked(mouse_mode_cycle_input, "cycle_mouse_mode"))
        sc = pm.get("shortcuts", DEFAULT_SHORTCUTS)

        hand_mode_cycle_input.setText(sc.get("cycle_hand_mode", "f1"))
        mouse_mode_cycle_input.setText(sc.get("cycle_mouse_mode", "f2"))
        
        hand_vectors_setting = QWidget()
        hand_vectors_setting.setFixedHeight(80)
        hand_vectors_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(hand_vectors_setting)
        
        hand_vectors_title = QLabel("Hand vectors:", hand_vectors_setting)
        hand_vectors_title.setGeometry(20,20,230, 40)
        hand_vectors_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = hand_vectors_title.font()
        f.setPointSize(16)
        hand_vectors_title.setFont(f)
        
        hand_vectors_options = QComboBox(hand_vectors_setting)
        hand_vectors_options.setGeometry(320,20,110,40)
        hand_vectors_options.setStyleSheet("""color: #030013; background: #e0dde5; border-radius: 8px; 
                                            
                                        """)
        
        hand_vectors_options.addItems([
            "On",
            "Off"
        ])
                
        for i in range(hand_vectors_options.count()):
            hand_vectors_options.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

        def _sync_modes_from_backend():
            # --- Hand mode ---
            resp = self.send_cmd("GET_HAND_MODE")
            # resp like: "OK multi_keyboard"
            if resp.startswith("OK"):
                mode = resp.split(maxsplit=1)[1].strip() if len(resp.split()) > 1 else ""
                backend_to_ui = {
                    "right": "Right pointer",
                    "left": "Left pointer",
                    "auto": "Auto",
                    "multi_keyboard": "MultiKB",
                }
                ui_txt = backend_to_ui.get(mode, "MultiKB")
                hand_mode_options.blockSignals(True)
                hand_mode_options.setCurrentText(ui_txt)
                hand_mode_options.blockSignals(False)

            # --- Mouse mode ---
            resp = self.send_cmd("GET_MOUSE_MODE")
            # resp like: "OK DISABLED"
            if resp.startswith("OK"):
                mode = resp.split(maxsplit=1)[1].strip() if len(resp.split()) > 1 else ""
                backend_to_ui = {
                    "DISABLED": "Disabled",
                    "CAMERA": "Camera",
                    "CURSOR": "Cursor",
                }
                ui_txt = backend_to_ui.get(mode, "Disabled")
                mouse_mode_options.blockSignals(True)
                mouse_mode_options.setCurrentText(ui_txt)
                mouse_mode_options.blockSignals(False)

        # call once before showing
        _sync_modes_from_backend()

        def _apply_hand_mode():
            txt = hand_mode_options.currentText().strip()
            # UI -> backend values
            ui_to_backend = {
                "Right pointer": "right",
                "Left pointer": "left",
                "Auto": "auto",
                "MultiKB": "multi_keyboard",
            }
            mode = ui_to_backend.get(txt, "auto")
            print("[UI] set hand mode ->", mode, self.send_cmd(f"SET_HAND_MODE {mode}"))

        def _apply_mouse_mode():
            txt = mouse_mode_options.currentText().strip()
            ui_to_backend = {
                "Disabled": "DISABLED",
                "Camera": "CAMERA",
                "Cursor": "CURSOR",
            }
            mode = ui_to_backend.get(txt, "DISABLED")
            print("[UI] set mouse mode ->", mode, self.send_cmd(f"SET_MOUSE_MODE {mode}"))

        def _apply_vectors():
            txt = hand_vectors_options.currentText().strip()
            mode = "on" if txt.lower() == "on" else "off"
            print("[UI] set vectors ->", mode, self.send_cmd(f"SET_VECTORS {mode}"))

        self._sync_mode_dropdowns(hand_mode_options, mouse_mode_options)
        hand_mode_options.currentTextChanged.connect(lambda _: _apply_hand_mode())
        mouse_mode_options.currentTextChanged.connect(lambda _: _apply_mouse_mode())
        hand_vectors_options.currentTextChanged.connect(lambda _: _apply_vectors())

        sensitivity_setting = QWidget()
        sensitivity_setting.setFixedHeight(80)
        sensitivity_setting.setStyleSheet("border: none; background: #252438; border-radius: 8px;")
        dialog_scroll_layout.addWidget(sensitivity_setting)
        
        sensitivity_title = QLabel("Sensitivity:", sensitivity_setting)
        sensitivity_title.setGeometry(20,20,230, 40)
        sensitivity_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = sensitivity_title.font()
        f.setPointSize(16)
        sensitivity_title.setFont(f)
        
        sensitivity_slider = QSlider(Qt.Horizontal, sensitivity_setting)
        sensitivity_slider.setGeometry(300, 33, 120, 22)
        sensitivity_slider.setMinimum(0)
        sensitivity_slider.setMaximum(100)
        saved = get_pm_setting("sensitivity", 50)
        sensitivity_slider.blockSignals(True)
        sensitivity_slider.setValue(int(saved))
        sensitivity_slider.blockSignals(False)
        def on_sens_changed(v: int):
            set_pm_setting("sensitivity", int(v))  # or float(v)/100 if you prefer float storage
            print("[UI] Saved sensitivity =", v)

            # Optional: tell backend immediately
            # If backend expects float:
            # self.send_cmd(f"SET_SENS {v/100:.2f}")
            self.send_cmd(f"SET_SENS {v}")
        sensitivity_slider.sliderReleased.connect(lambda: on_sens_changed(sensitivity_slider.value()))

        _apply_hand_mode()
        dialog.exec()

    # main command sender
    def send_cmd(self, cmd: str):
        try:
            with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=3.0) as s:
                s.settimeout(1.0)
                s.sendall((cmd.strip() + "\n").encode("utf-8"))

                data = s.recv(1024)
                if not data:
                    return "ERR: no response"
                return data.decode("utf-8", errors="ignore").strip()

        except Exception as e:
            return f"ERR: {e}"
        
    def tab_rename(self, index):
        if (index == 0):
            return
        
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Rename Tab")
        dialog.setLabelText("Enter new name:")
        dialog.setFixedSize(300,300)
        dialog.setStyleSheet("QDialog {background: #3c384d;} QLabel {color: #e0dde5; font-size: 14px;}")
        
        ok = dialog.exec()
        text = dialog.textValue()

        if not ok or not text:
            return

        current_name = self.tabs.tabText(index)

        # sanitize filename + profile ID
        new_name = text.strip().replace(" ", "_")

        if not new_name:
            QMessageBox.warning(self, "Invalid name", "Profile name cannot be empty.")
            return

        # Optional: prevent renaming to the same name
        if new_name == current_name:
            return

        # Rename JSON file + update profileManager.json
        success = self.profiles.renameProfile(current_name, new_name)

        if not success:
            QMessageBox.warning(self, "Rename failed", "Could not rename profile.")
            return

        # Update tab label
        self.tabs.setTabText(index, new_name)
        self._on_tab_changed(index)


    def _on_tab_changed(self, idx: int):
        if self.tabs.tabBar().tabData(idx) == "add_tab_button":
            return

        profile_id = self._profile_id_for_tab(idx)
        if not profile_id:
            return

        resp = self.send_cmd(f"SET_PROFILE {profile_id}")
        print("[UI] SET_PROFILE", profile_id, "->", resp)



    #function for the new tab button
    def new_tab_button(self, index):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        library_path = os.path.join(BASE_DIR, "resource", "Definition-Search-Book--Streamline-Core.png")
        plus_path = os.path.join(BASE_DIR, "resource", "Add-Circle--Streamline-Core.png")
        camera_path = os.path.join(BASE_DIR, "resource", "Camera-1--Streamline-Core.png")
        setting_path = os.path.join(BASE_DIR, "resource", "Cog--Streamline-Core.png")
        #check if the tab clicked is the last tab
        if self.tabs.tabBar().tabData(index) == "add_tab_button":
            # 1) create profile id + json file
            new_profile_id = self._generate_new_profile_id()
            ok = self._create_profile_file(new_profile_id)
            if not ok:
                QMessageBox.warning(self, "Create Profile Failed", "Could not create profile JSON file.")
                return

            # 2) build tab UI (buttons + scroll) in ONE place
            tab_index = self._add_profile_tab(new_profile_id)

            # 3) switch to it
            self.tabs.setCurrentIndex(tab_index)

            # 4) tell backend
            self._on_tab_changed(tab_index)
            return


    def _add_profile_tab(self, profile_id: str) -> int:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        library_path = os.path.join(BASE_DIR, "resource", "Definition-Search-Book--Streamline-Core.png")
        plus_path    = os.path.join(BASE_DIR, "resource", "Add-Circle--Streamline-Core.png")
        camera_path  = os.path.join(BASE_DIR, "resource", "Camera-1--Streamline-Core.png")
        setting_path = os.path.join(BASE_DIR, "resource", "Cog--Streamline-Core.png")
        pencil_path = os.path.join(BASE_DIR, "resource", "Pencil--Streamline-Core.png")

        # needed for resize purpose
        geom = self.geometry()
        x = (geom.width() - 1698) // 2 + 99

        # Insert new tab BEFORE the "+" tab
        insert_index = self.last_tab_index

        new_tab = QWidget()
        new_tab.setStyleSheet("""
            QWidget {
                background-color: rgb(60, 56, 77);
            }
        """)

        # ---- 4 buttons container ----
        new_four_buttons_container = QWidget(new_tab)
        new_four_buttons_container.setGeometry(1178, 40, 470, 80)
        new_four_buttons_container.move(max(geom.width() - new_four_buttons_container.width()-50, 0),
                                        new_four_buttons_container.y())
        new_four_buttons_container.setProperty("4_buttons_container", True)

        library_button = QPushButton(new_four_buttons_container)
        library_button.setGeometry(0,0,80,80)
        library_button.setIcon(QIcon(library_path))
        library_button.setIconSize(QSize(50, 50))
        library_button.setFlat(True)
        library_button.clicked.connect(self.on_library_clicked)

        plus_button = QPushButton(new_four_buttons_container)
        plus_button.setGeometry(130,0,80,80)
        plus_button.setIcon(QIcon(plus_path))
        plus_button.setIconSize(QSize(50,50))
        plus_button.setFlat(True)
        plus_button.clicked.connect(self.new_gesture_button_function)

        camera_button = QPushButton(new_four_buttons_container)
        camera_button.setGeometry(260, 0, 80,80)
        camera_button.setIcon(QIcon(camera_path))
        camera_button.setIconSize(QSize(50,50))
        camera_button.setFlat(True)
        camera_button.clicked.connect(self.on_camera_clicked)

        setting_button = QPushButton(new_four_buttons_container)
        setting_button.setGeometry(390, 0, 80, 80)
        setting_button.setIcon(QIcon(setting_path))
        setting_button.setIconSize(QSize(50,50))
        setting_button.setFlat(True)
        setting_button.clicked.connect(self.setting_dialog)

        # ---- scroll area ----
        scroll_container = QScrollArea(new_tab)
        scroll_container.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_container.setGeometry(99,150,1599, 962)
        scroll_container.setGeometry(max(x,0), scroll_container.y(),
                                    max(geom.width()-scroll_container.x(),0),
                                    max(962+geom.height()-1183, 0))
        scroll_container.setStyleSheet("border: none;")
        scroll_container.setProperty("individual_sub_bar_container", True)

        scroll_content = QWidget()
        scroll_container.setWidget(scroll_content)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)
        scroll_container.setWidgetResizable(True)

        # ---- insert the tab ----
        self.tabs.insertTab(insert_index, new_tab, QIcon(pencil_path), profile_id)

        # keep "+" tab at the end
        self.last_tab_index = self.tabs.count() - 1
        self.tabs.tabBar().setTabData(self.last_tab_index, "add_tab_button")
        self.tabs.tabBar().setTabText(self.last_tab_index, "")
        self.tabs.tabBar().setTabButton(self.last_tab_index, QTabBar.RightSide, None)

        return insert_index


    def _create_new_profile_file(self, profile_id: str):
        path = self._profile_path(profile_id)
        if os.path.exists(path):
            return False
        data = {"Profile_ID": profile_id, "Actions": []}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True

    def close_tab(self, index):
        # Never close the first tab
        if index == 0:
            return

        # Never close the "+" tab
        if self.tabs.tabBar().tabData(index) == "add_tab_button":
            return

        profile_name = self.tabs.tabText(index)

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Delete profile '{profile_name}'?\nThis will permanently delete its JSON file.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete JSON file + update profileManager.json
        success = self.profiles.deleteProfile(profile_name)
        if not success:
            QMessageBox.warning(
                self,
                "Delete failed",
                f"Failed to delete profile '{profile_name}'."
            )
            return

        # --- UI tab removal (existing logic) ---
        current = self.tabs.currentIndex()
        next_index = None

        if index == current:
            next_index = index - 1
            if self.tabs.tabBar().tabData(next_index) == "add_tab_button":
                next_index = max(0, next_index - 1)

        self.tabs.removeTab(index)

        self.last_tab_index = self.tabs.count() - 1

        if next_index is not None:
            if index < next_index:
                next_index -= 1
            self.tabs.setCurrentIndex(max(0, min(next_index, self.tabs.count() - 2)))
            self._on_tab_changed(self.tabs.currentIndex())



    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
    
    def _reload_active_tab_actions(self):
        idx = self.tabs.currentIndex()
        if self.tabs.tabBar().tabData(idx) == "add_tab_button":
            return

        profile_id = self._profile_id_for_tab(idx)
        if not profile_id:
            return

        tab = self.tabs.widget(idx)
        if tab is None:
            return

        scroll = tab.findChild(QScrollArea)
        if scroll is None or scroll.property("individual_sub_bar_container") is not True:
            return

        content = scroll.widget()
        layout = content.layout()
        if layout is None:
            return

        # 1) clear UI rows
        self._clear_layout(layout)

        # 2) rebuild rows
        if profile_id == "Default":
            data = self._load_profile_json("Default")
            actions = data.get("Actions", [])
            if not isinstance(actions, list):
                actions = []

            class _ActShim:
                def __init__(self, d): self._d = d
                def getGName(self): return self._d.get("G_name")
                def getKeyPressed(self): return self._d.get("key_pressed")
                def getInputType(self): return self._d.get("input_type")
                def getName(self): return self._d.get("name")

            for row in actions:
                if isinstance(row, dict):
                    self.build_action_row(layout, profile_id="Default", act=_ActShim(row))
        else:
            current_profile = self.profiles.loadProfile(profile_id)
            if current_profile is None:
                print("[UI] reload tab: profile not found:", profile_id)
                return
            for act in (current_profile.getActionList() or []):
                self.build_action_row(layout, profile_id=profile_id, act=act)


        # 3) ensure backend uses latest mappings for this profile
        resp = self.send_cmd(f"SET_PROFILE {profile_id}")
        print("[UI] Reloaded active tab + SET_PROFILE ->", resp)

                
    def on_library_clicked(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        rec_path   = os.path.join(BASE_DIR, "resource", "Webcam-Video-Circle--Streamline-Core.png")
        trash_path = os.path.join(BASE_DIR, "resource", "Recycle-Bin-2--Streamline-Core.png")

        # ---- locate GestureList.json ----
        # If designerapp is under src/designerapp, and GestureList.json is under src/
        SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # go up from designerapp
        gesturelist_path = os.path.join(SRC_DIR, "GestureList.json")

        print("[UI] GestureList path =", gesturelist_path)
        print("[UI] Exists? =", os.path.exists(gesturelist_path))

        def load_gestures():
            if not os.path.exists(gesturelist_path):
                return []
            try:
                with open(gesturelist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # keep only non-empty strings
                    out = []
                    seen = set()
                    for g in data:
                        if isinstance(g, str):
                            g2 = g.strip()
                            if g2 and g2 not in seen:
                                out.append(g2)
                                seen.add(g2)
                    return out
            except Exception as e:
                print("[UI] Failed to load GestureList.json:", e)
            return []

        def save_gestures(lst):
            try:
                with open(gesturelist_path, "w", encoding="utf-8") as f:
                    json.dump(lst, f, indent=4)
                return True
            except Exception as e:
                print("[UI] Failed to save GestureList.json:", e)
                return False

        gestures = load_gestures()

        dialog = QDialog(self)
        dialog.setWindowTitle("Action Library")
        dialog.setFixedSize(400, 500)
        dialog.setModal(True)

        top_frame = QFrame(dialog)
        top_frame.setGeometry(0, 0, 400, 70)
        top_frame.setStyleSheet("background-color: #030013;")

        label_title = QLabel("Action Library", top_frame)
        label_title.setGeometry(20, 20, 250, 31)
        label_title.setStyleSheet("color: #e0dde5; background: transparent;")
        f = label_title.font()
        f.setPointSize(14)
        label_title.setFont(f)

        # ---- Selected gesture state ----
        selected = {"name": None}

        def set_selected(name: str):
            selected["name"] = name
            # optional: update title or status
            if name:
                label_title.setText(f"Action Library  ({name})")
            else:
                label_title.setText("Action Library")

        # ---- record button (uses selected gesture) ----
        rec_button = QPushButton(top_frame)
        rec_button.clicked.connect(self.new_gesture_dialog)
        rec_button.setGeometry(340, 10, 50, 50)
        rec_button.setIcon(QIcon(rec_path))
        rec_button.setIconSize(QSize(45, 45))
        rec_button.setStyleSheet("border: none; background: transparent;")

        dialog_scroll = QScrollArea(dialog)
        dialog_scroll.setGeometry(0, 70, 400, 430)
        dialog_scroll.setStyleSheet("background: #3c384d; border: none;")
        dialog_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        dialog_scroll_content = QWidget()
        dialog_scroll.setWidget(dialog_scroll_content)
        dialog_scroll_layout = QVBoxLayout(dialog_scroll_content)
        dialog_scroll_layout.setAlignment(Qt.AlignTop)
        dialog_scroll_layout.setContentsMargins(10, 30, 10, 10)
        dialog_scroll.setWidgetResizable(True)

        
        def rebuild_rows():
            # clear
            while dialog_scroll_layout.count():
                item = dialog_scroll_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            gs = load_gestures()

            if not gs:
                empty = QLabel("No gestures found in GestureList.json")
                empty.setStyleSheet("color: #e0dde5; background: transparent;")
                empty.setAlignment(Qt.AlignCenter)
                dialog_scroll_layout.addWidget(empty)
                set_selected(None)
                return

            for gname in gs:
                # row container
                gesture_bar = QWidget()
                gesture_bar.setFixedHeight(60)
                gesture_bar.setStyleSheet("border: none; background: transparent;")
                dialog_scroll_layout.addWidget(gesture_bar)

                # clickable frame (select)
                gesture_frame = QFrame(gesture_bar)
                gesture_frame.setGeometry(0, 0, 290, 50)
                gesture_frame.setStyleSheet("border: none; background: #252438; border-radius: 10px;")

                gesture_text = QLabel(gname, gesture_frame)
                gesture_text.setGeometry(10, 10, 270, 30)
                gesture_text.setStyleSheet("border: none; background: transparent; color: #e0dde5;")
                gesture_text.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                f = gesture_text.font()
                f.setPointSize(14)
                gesture_text.setFont(f)

                # Make the row selectable by click
                def make_click_handler(name):
                    def _():
                        set_selected(name)
                    return _

                select_btn = QPushButton(gesture_bar)
                select_btn.setGeometry(0, 0, 290, 50)
                select_btn.setStyleSheet("background: transparent; border: none;")
                select_btn.clicked.connect(make_click_handler(gname))

                # trash button (optional delete)
                trash_button = QPushButton(gesture_bar)
                trash_button.setGeometry(310, 5, 40, 40)
                trash_button.setIcon(QIcon(trash_path))
                trash_button.setIconSize(QSize(30, 30))
                trash_button.setFlat(True)

                def make_delete_handler(name):
                    def _():
                        reply = QMessageBox.question(
                            dialog,
                            "Delete gesture",
                            f"Delete '{name}' from GestureList.json?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply != QMessageBox.Yes:
                            return
                        cur = load_gestures()
                        cur2 = [x for x in cur if x != name]
                        if save_gestures(cur2):
                            # if deleting selected
                            if selected["name"] == name:
                                set_selected(None)
                                resp = self.send_cmd(f"DELETE_GESTURE {name}")
                                print("[UI] DELETE_GESTURE resp:", resp)
                            rebuild_rows()
                    return _

                trash_button.clicked.connect(make_delete_handler(gname))

        rebuild_rows()
        # ---------------- AUTO REFRESH (file watcher via polling) ----------------
        dialog._is_rebuilding = False

        # initialize last_mtime to the current file mtime
        try:
            dialog._last_mtime = os.path.getmtime(gesturelist_path) if os.path.exists(gesturelist_path) else None
        except Exception:
            dialog._last_mtime = None

        def _tick_refresh():
            if dialog._is_rebuilding:
                return
            try:
                mtime = os.path.getmtime(gesturelist_path) if os.path.exists(gesturelist_path) else None
                if mtime != dialog._last_mtime:
                    dialog._last_mtime = mtime
                    dialog._is_rebuilding = True
                    rebuild_rows()
                    dialog._is_rebuilding = False
            except Exception:
                dialog._is_rebuilding = False

        timer = QTimer(dialog)
        timer.timeout.connect(_tick_refresh)
        timer.start(500)
        dialog.exec()

        
    def _start_record_from_library(self, library_dialog: QDialog):
        # optional: close library so user focuses on the create/record flow
        try:
            library_dialog.accept()
        except Exception:
            pass

        # start your existing gesture input flow
        self.new_gesture_dialog()

    def new_gesture_button_function(self):
        # This plus button should ADD a new empty row at bottom
        current_index = self.tabs.currentIndex()
        profile_id = self.tabs.tabText(current_index).strip()

        # safety: ignore "+" tab
        if self.tabs.tabBar().tabData(current_index) == "add_tab_button":
            return

        if not profile_id:
            return

        self._append_empty_action_mapping(profile_id)

    
    def new_gesture_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Gesture")
        dialog.setFixedSize(300, 220)
        dialog.setModal(True)
        dialog.setStyleSheet("background: #3c384d;")
        layout = QVBoxLayout(dialog)

        title = QTextEdit("Gesture Name")
        title.setAlignment(Qt.AlignLeft)
        title.setReadOnly(True)
        font = title.font()
        font.setPointSize(16)
        title.setFont(font)
        title.setStyleSheet("border: none;" 
            "color: #ffffff")
        layout.addWidget(title)

        gesture_name_box = QTextEdit("")
        gesture_name_box.setAlignment(Qt.AlignCenter)
        font2 = gesture_name_box.font()
        font2.setPointSize(14)
        gesture_name_box.setFont(font2)
        gesture_name_box.setStyleSheet("""
            background-color: #e0dde5;f
            color: #030013;
        """)
        layout.addWidget(gesture_name_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setStyleSheet("""
        QPushButton {
            background-color: #e0dde5;
            color: #030013;
            border-radius: 6px;
            padding: 6px;
        }
        QPushButton:hover {
            background-color: #cfcbd6;
        }
        """)
        layout.addWidget(buttons)

        def on_ok():
            self._refresh_action_dropdowns_from_gesturelist()
            gname = gesture_name_box.toPlainText().strip()
            if not gname:
                QMessageBox.warning(self, "Error", "Gesture name cannot be empty.")
                return

            resp = self.send_cmd(f"CREATE_GESTURE {gname}")
            if resp.startswith("OK"):
                # update dropdowns immediately + reload current tab rows
                self._refresh_action_dropdowns_from_gesturelist()
                self._reload_active_tab_actions()
                dialog.accept()
            else:
                QMessageBox.critical(self, "Create Gesture Failed", resp)


        buttons.accepted.connect(on_ok)
        buttons.rejected.connect(dialog.reject)

        dialog.exec()

    
    #for deleting new gesture
    def trash_button_function():
        return None
    # for the main power button        
    def main_power_button(self):
        try:
            self.send_cmd("QUIT")
        except Exception:
            pass

        if getattr(self, "_backend_proc", None) is not None and self._backend_proc.poll() is None:
            self._backend_proc.terminate()

        QApplication.quit()

        
        
    # some widgets in the ui has dynamic property, kinda like tag in unity, this function is to find all widgets with the same dynamic property    
    def find_widgets_with_property(self, root: QWidget, prop_name):
        result = []
        for w in root.findChildren(QWidget):
            if w.property(prop_name) is True:
                result.append(w)
        return result    
    
    #this function is for moving widgets around when the window resize
    def resizeEvent(self, event):
        self.window.setGeometry(self.rect())
        geom = self.geometry()
        x = (geom.width() - 1698) // 2 + 99
        
        self.power_button.move(geom.width()-self.power_button.width()-10, 10)
        
        self.tabs.setGeometry(0, 20, geom.width(), geom.height()-20)
        
        individual_bars = self.find_widgets_with_property(self.window, "individual_sub_bar_container")
        
        buttons_containers = self.find_widgets_with_property(self.window, "4_buttons_container")
        
        for i in individual_bars:
            i.setGeometry(max(x,0), i.y(), max(geom.width()-i.x(),0), max(962+geom.height()-1183, 0))
            
        for b in buttons_containers:
            b.move(max(geom.width() - b.width()-40, 0), b.y())
        
        super().resizeEvent(event)

    def _profile_path(self, profile_id: str) -> str:
        # profiles live in src (parent of designerapp)
        if profile_id == "Default":
            return str(self.PARENT_DIR / "Default.json")
        return str(self.PARENT_DIR / f"profile_{profile_id}.json")

    
    def _sanitize_profile_id(self, name: str) -> str:
        # Keep it simple: spaces -> underscore, strip
        out = (name or "").strip().replace(" ", "_")
        # optional extra safety: remove weird chars
        out = "".join(ch for ch in out if ch.isalnum() or ch in ("_", "-"))
        return out

    def _profile_exists(self, profile_id: str) -> bool:
        return os.path.exists(self._profile_path(profile_id))

    def _generate_new_profile_id(self) -> str:
        # Generates Profile_1, Profile_2, ...
        i = 1
        while True:
            pid = f"Profile_{i}"
            if not self._profile_exists(pid):
                return pid
            i += 1

    def _create_profile_file(self, profile_id: str) -> bool:
        try:
            path = self._profile_path(profile_id)
            if os.path.exists(path):
                return False
            data = {"Profile_ID": profile_id, "Actions": []}
            self._save_profile_json(profile_id, data)
            return True
        except Exception as e:
            print("[UI] Failed to create profile file:", e)
            return False


    def _load_profile_json(self, profile_id: str) -> dict:
        path = self._profile_path(profile_id)
        if not os.path.exists(path):
            return {"Profile_ID": profile_id, "Actions": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_profile_json(self, profile_id: str, data: dict) -> None:
        path = self._profile_path(profile_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _update_action_in_profile(self, profile_id: str, action_name: str, new_fields: dict):
        data = self._load_profile_json(profile_id)
        actions = data.get("Actions", [])
        if not isinstance(actions, list):
            actions = []
            data["Actions"] = actions

        found = False
        for a in actions:
            if isinstance(a, dict) and a.get("name") == action_name:
                a.update(new_fields)
                found = True
                break

        if not found:
            # create new entry if missing
            row = {"name": action_name, "G_name": None, "key_pressed": None, "input_type": None, "key_type": None}
            row.update(new_fields)
            actions.append(row)

        data["Profile_ID"] = profile_id
        self._save_profile_json(profile_id, data)

    def save_action_edit(self, profile_id, action_name, new_gname, new_key, new_input_type, new_name=None):
        # profile_<id>.json is beside ProfileManager.py, so use ProfileManager's base_dir for correct pathing
        profiles = ProfileManager()
        base_dir = os.path.dirname(os.path.abspath(__import__("ProfileManager").__file__))
        profile_path = self._profile_path(profile_id)

        if not os.path.exists(profile_path):
            print("[UI] save_action_edit: profile file missing:", profile_path)
            return

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            actions = data.get("Actions", [])
            if not isinstance(actions, list):
                print("[UI] save_action_edit: Actions is not a list")
                return

            # find by action 'name' (unique identifier)
            for a in actions:
                if not isinstance(a, dict):
                    continue

                if a.get("name") == action_name:
                    # update G_name
                    a["G_name"] = (new_gname or "").strip()

                    # update key_pressed
                    nk = (new_key or "").strip()
                    a["key_pressed"] = nk if nk != "" else None

                    # update input_type (convert "Double Click" back if you want)
                    it = (new_input_type or "").strip()
                    if it == "Double Click":
                        it = "D_Click"
                    a["input_type"] = it

                    # rename only AFTER match
                    if new_name is not None:
                        nn = new_name.strip()
                        if nn != "":
                            a["name"] = nn

                    break
            else:
                print(f"[UI] save_action_edit: action '{action_name}' not found in profile_{profile_id}.json")
                return



            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"[UI] Saved edit -> profile_{profile_id}.json : {action_name}")

        except Exception as e:
            print("[UI] save_action_edit failed:", e)



    def on_camera_clicked(self):
        print(self.send_cmd("TOGGLE_CAMERA"))

    def on_setting_clicked(self):
        print(self.send_cmd("TOGGLE_GUI"))

    def _gesturelist_path(self) -> str:
        # GestureList.json is in src (parent of designerapp)
        return os.path.join(str(self.PARENT_DIR), "GestureList.json")

    def _load_gesture_list(self) -> list[str]:
        path = self._gesturelist_path()
        out = []
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    out = [x.strip() for x in data if isinstance(x, str) and x.strip()]
        except Exception as e:
            print("[UI] Failed loading GestureList.json:", e)

        # optional: dedupe while keeping order
        seen = set()
        deduped = []
        for g in out:
            if g not in seen:
                deduped.append(g)
                seen.add(g)
        return deduped

    def build_action_row(self, current_scroll_layout, profile_id: str, act):
        """
        Reusable UI row builder.
        - 'name' (action name) is editable (gesture_edit)
        - 'G_name' (gesture) is dropdown (action_box)
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        pencil_path = os.path.join(BASE_DIR, "resource", "Pencil--Streamline-Core.png")
        
        g_name = act.getGName()
        key    = act.getKeyPressed()
        itype  = act.getInputType()
        a_name = act.getName()
        action_id_ref = {"id": a_name}

        sub_bar_widget = QWidget()
        sub_bar_widget.setFixedHeight(150)
        current_scroll_layout.addWidget(sub_bar_widget)

        sub_bar_frame = QFrame(sub_bar_widget)
        sub_bar_frame.setGeometry(0, 0, 1400, 125)
        sub_bar_frame.setProperty("individual_sub_bar", True)
        sub_bar_frame.setStyleSheet("""
            QFrame[individual_sub_bar] {
                background-color: #252438;
                border-radius: 12px;
            }
        """)
        
                
        pencil_icon = QLabel(sub_bar_frame)
        pencil_icon.setGeometry(60,50,25,25)
        pencil_icon.setStyleSheet("background: transparent; border: none;")
        pencil_icon.setPixmap(QPixmap(pencil_path))
        pencil_icon.setScaledContents(True)

        # ----- Editable NAME field (this edits "name") -----
        gesture_edit = QLineEdit(a_name, sub_bar_frame)
        gesture_edit.setGeometry(90, 42, 180, 40)
        gesture_edit.setAlignment(Qt.AlignLeft)
        gFont = gesture_edit.font()
        gFont.setBold(True)
        gFont.setPointSize(16)
        gFont.setPointSize(14)
        gesture_edit.setFont(gFont)
        gesture_edit.setStyleSheet("""
            border: none;
            background: transparent;
            color: rgb(224, 221, 229);
        """)

        # ----- Key label -----
        key_input = QTextEdit("KEY INPUT", sub_bar_frame)
        key_input.setGeometry(350, 25, 120, 30)
        key_input.setAlignment(Qt.AlignCenter)
        kFont = key_input.font()
        kFont.setBold(True)
        kFont.setPointSize(10)
        key_input.setFont(kFont)
        key_input.setReadOnly(True)
        key_input.setStyleSheet("""
            border: none;
            color: rgb(224, 221, 229);
            background: transparent;
        """)

        # ----- KEY capture button (replaces text box) -----
        key_btn = QPushButton(sub_bar_frame)
        key_btn.setGeometry(325, 55, 150, 40)

        def _set_key_btn_text(val: str | None):
            if val is None or str(val).strip() == "":
                key_btn.setText("Set Key")
            else:
                key_btn.setText(str(val).strip())

        _set_key_btn_text(key)

        key_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(224, 221, 229);
                color: rgb(0, 0, 0);
                border: none;
                border-radius: 6px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgb(200, 198, 205);
            }
            QPushButton:pressed {
                background-color: rgb(180, 178, 185);
            }
        """)


        # ----- Input type label -----
        input_type = QTextEdit("INPUT TYPE", sub_bar_frame)
        input_type.setGeometry(643, 25, 120, 30)
        input_type.setAlignment(Qt.AlignCenter)
        iFont = input_type.font()
        iFont.setBold(True)
        iFont.setPointSize(10)
        input_type.setFont(iFont)
        input_type.setReadOnly(True)
        input_type.setStyleSheet("""
            border: none;
            color: rgb(224, 221, 229);
            background: transparent;
        """)

        # ----- Input type dropdown (edits "input_type") -----
        input_type_box = QComboBox(sub_bar_frame)
        input_type_box.setGeometry(625, 55, 150, 40)
        iFont2 = input_type_box.font()
        iFont2.setPointSize(12)
        input_type_box.setFont(iFont2)
        input_type_box.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)
        input_type_box.addItems(["Click", "Hold", "Double Click"])
        for i in range(input_type_box.count()):
            input_type_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

        # normalize "D_Click" -> "Double Click" for display
        itype_norm = (itype or "").strip()
        if itype_norm == "D_Click":
            itype_norm = "Double Click"
        idx = input_type_box.findText(itype_norm)
        if idx >= 0:
            input_type_box.setCurrentIndex(idx)

        # ----- Divider line -----
        line = QFrame(sub_bar_frame)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setLineWidth(1)
        line.setGeometry(937, 20, 3, 80)
        line.setStyleSheet("background-color: rgb(224, 221, 229);")

        # ----- ACTION label (your UI label) -----
        action_label = QTextEdit("ACTION", sub_bar_frame)
        action_label.setGeometry(1125, 25, 150, 30)
        action_label.setAlignment(Qt.AlignCenter)
        aFont = action_label.font()
        aFont.setBold(True)
        aFont.setPointSize(10)
        action_label.setFont(aFont)
        action_label.setReadOnly(True)
        action_label.setStyleSheet("""
            border: none;
            color: rgb(224, 221, 229);
            background: transparent;
        """)

        # ----- Dropdown for G_name (gesture binding) -----
        action_box = QComboBox(sub_bar_frame)
        action_box.setGeometry(1100, 55, 200, 40)
        action_box.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)
        action_box.setProperty("is_action_box", True)
        gesture_list = self._load_gesture_list()
        action_box.clear()
        action_box.addItems(gesture_list)
        aFont2 = action_box.font()
        aFont2.setPointSize(12)
        action_box.setFont(aFont2)
        for i in range(action_box.count()):
            action_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

        # select current g_name (G_name)
        idx2 = action_box.findText(g_name)
        if idx2 >= 0:
            action_box.setCurrentIndex(idx2)
        elif g_name:
            # if current g_name not in GestureList, still show it
            action_box.insertItem(0, g_name)
            action_box.setCurrentIndex(0)

        # ----- Trash button -----
        trash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "Recycle-Bin-2--Streamline-Core.png")
        trash_button = QPushButton(sub_bar_widget)
        trash_button.setGeometry(1450, 20, 80, 80)
        trash_button.setIcon(QIcon(trash_path))
        trash_button.setIconSize(QSize(50, 50))
        trash_button.setFlat(True)
        trash_button.setStyleSheet("""
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.08);
                border-radius: 6px
            }
        """)
        def on_trash_clicked():
            # 1) delete from JSON
            try:
                data = self._load_profile_json(profile_id)
                actions = data.get("Actions", [])
                if not isinstance(actions, list):
                    actions = []

                target = action_id_ref["id"]
                new_actions = []
                for a in actions:
                    if isinstance(a, dict) and a.get("name") == target:
                        continue
                    new_actions.append(a)

                data["Actions"] = new_actions
                self._save_profile_json(profile_id, data)
                print(f"[UI] Deleted action row '{target}' from profile_{profile_id}.json")
            except Exception as e:
                print("[UI] Failed to delete row:", e)
                return

            # 2) remove from UI
            sub_bar_widget.setParent(None)
            sub_bar_widget.deleteLater()

        trash_button.clicked.connect(on_trash_clicked)


        # ----- ONE commit_change (wired AFTER action_box exists) -----
        def commit_change():
            old_id = action_id_ref["id"]
            new_id = gesture_edit.text().strip()

            self.save_action_edit(
                profile_id=profile_id,
                action_name=old_id,                          # stable lookup (current id)
                new_gname=action_box.currentText().strip(),   # writes to G_name
                new_key=key_btn.text().strip() if key_btn.text().strip().lower() != "set key" else "",
                new_input_type=input_type_box.currentText(),
                new_name=new_id                               # writes to name
            )

            # if rename succeeded logically, update our local id so future edits & delete work
            if new_id and new_id != old_id:
                action_id_ref["id"] = new_id

        def on_key_button_clicked():
            dlg = KeyCaptureDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return

            # captured_key meanings:
            # None  -> cancelled
            # ""    -> NULL
            # "a"   -> real key
            val = dlg.captured_key

            if val is None:
                return

            if val == "":
                _set_key_btn_text("NULL")
            else:
                _set_key_btn_text(val)

            commit_change()


        key_btn.clicked.connect(on_key_button_clicked)



        gesture_edit.editingFinished.connect(commit_change)
        input_type_box.currentTextChanged.connect(lambda _: commit_change())
        action_box.currentTextChanged.connect(lambda _: commit_change())

    def _append_empty_action_mapping(self, profile_id: str):
        """
        Add a new empty mapping row at the bottom of the current tab's scroll layout,
        and also append an empty action entry into profile_<id>.json.

        - name: auto-generated unique (e.g. "new_action_1")
        - G_name/key_pressed/input_type/key_type: None
        """
        # ---- find the current tab scroll layout ----
        current_index = self.tabs.currentIndex()
        tab = self.tabs.widget(current_index)
        if tab is None:
            return

        scroll = tab.findChild(QScrollArea)
        if scroll is None:
            return

        content = scroll.widget()
        layout = content.layout()
        if layout is None:
            return

        # ---- load profile json ----
        path = self._profile_path(profile_id)
        data = self._load_profile_json(profile_id)
        actions = data.get("Actions", [])
        if not isinstance(actions, list):
            actions = []
            data["Actions"] = actions

        # ---- generate a unique default action name ----
        existing_names = set()
        for a in actions:
            if isinstance(a, dict):
                n = a.get("name")
                if isinstance(n, str):
                    existing_names.add(n)

        base = "new_action"
        i = 1
        new_name = f"{base}_{i}"
        while new_name in existing_names:
            i += 1
            new_name = f"{base}_{i}"

        # ---- append empty row to json ----
        new_row = {
            "name": new_name,
            "G_name": None,
            "key_pressed": None,
            "input_type": None,
            "key_type": None
        }
        actions.append(new_row)
        data["Profile_ID"] = profile_id
        self._save_profile_json(profile_id, data)

        # ---- build UI row using an Actions-like object (minimal shim) ----
        class _ActShim:
            def __init__(self, d):
                self._d = d
            def getGName(self): return self._d.get("G_name")
            def getKeyPressed(self): return self._d.get("key_pressed")
            def getInputType(self): return self._d.get("input_type")
            def getName(self): return self._d.get("name")

        self.build_action_row(layout, profile_id=profile_id, act=_ActShim(new_row))

    def _tick_gesturelist_refresh(self):
        if getattr(self, "_gesturelist_refreshing", False):
            return

        try:
            path = self._gesturelist_path()
            mtime = os.path.getmtime(path) if os.path.exists(path) else None

            if mtime != getattr(self, "_gesturelist_last_mtime", None):
                self._gesturelist_last_mtime = mtime
                self._gesturelist_refreshing = True
                self._refresh_action_dropdowns_from_gesturelist()
                self._gesturelist_refreshing = False

        except Exception as e:
            self._gesturelist_refreshing = False
            print("[UI] GestureList watcher error:", e)


    def _refresh_action_dropdowns_from_gesturelist(self):
        gesture_list = self._load_gesture_list()  # you already wrote this helper

        # Find every action dropdown you created and refresh it
        for tab_index in range(self.tabs.count()):
            if self.tabs.tabBar().tabData(tab_index) == "add_tab_button":
                continue

            tab = self.tabs.widget(tab_index)
            if tab is None:
                continue

            for cb in tab.findChildren(QComboBox):
                if cb.property("is_action_box") is True:
                    current = cb.currentText()

                    # block signals so it doesn't commit_change while repopulating
                    was_blocked = cb.blockSignals(True)

                    cb.clear()
                    cb.addItems(gesture_list)

                    # restore selection
                    idx = cb.findText(current)
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
                    elif current:
                        cb.insertItem(0, current)
                        cb.setCurrentIndex(0)

                    cb.blockSignals(was_blocked)





# this class is only for implementing the text box that will capture the user keyboard input when adding new gesture        
class KeyCaptureBox(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setReadOnly(True)
        self.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.capturing = False

    def mousePressEvent(self, event):
        self.setText("Press a key...")
        self.capturing = True
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self.capturing:
            self.setFocusPolicy(Qt.StrongFocus)
            self.setFocus()

            key = event.key()
            text = event.text()

            # Handle special keys nicely
            if key == Qt.Key_Space:
                key_name = "space"
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                key_name = "enter"
            elif key == Qt.Key_Shift:
                key_name = "shift"
            elif key == Qt.Key_Control:
                key_name = "ctrl"
            elif key == Qt.Key_Alt:
                key_name = "alt"
            elif key == Qt.Key_Backspace:
                key_name = "backspace"
            elif key == Qt.Key_Tab:
                key_name = "tab"
            elif key == Qt.Key_Escape:
                key_name = "esc"
            elif key == Qt.Key_Up:
                key_name = "up"
            elif key == Qt.Key_Down:
                key_name = "down"
            elif key == Qt.Key_Left:
                key_name = "left"
            elif key == Qt.Key_Right:
                key_name = "right"
            elif key == Qt.Key_Delete:
                key_name = "delete"
            else:
                # letters/numbers/symbols
                key_name = text.lower().strip() if text else f"key_{key}"

            self.setText(key_name)
            self.capturing = False
            event.accept()
            return

        super().keyPressEvent(event)

def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    run()
