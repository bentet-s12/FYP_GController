from . import resources_rc
import keyboard
import subprocess
import sys
import os
import json
import threading
import socket
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice, QTimer
from PySide6.QtGui import QPixmap, QIcon, QFont, QKeySequence
from PySide6.QtCore import QSize
from PySide6.QtWidgets import ( 
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QStatusBar, QMessageBox, QLabel, QPushButton, 
    QLineEdit, QComboBox, QTabBar, QToolButton, QDialog, QScrollArea, 
    QSizePolicy, QFrame, QTextBrowser, QGraphicsDropShadowEffect, QTabWidget, QTextEdit, QDialogButtonBox
)
from pathlib import Path
from ProfileManager import ProfileManager

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 50555

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

    
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.ui")
        file = QFile(ui_path)
        if not file.open(QIODevice.ReadOnly):
            raise RuntimeError(f"Unable to open UI file: {ui_path}")
        loader = QUiLoader()
        self.window = loader.load(file, self)
        file.close()
        if self.window is None:
            raise RuntimeError(loader.errorString())
        
        
        # self.profiles = ProfileManager()
        #self.tabs refer to the entire tab widget not just the tab bar
        self.tabs = self.window.findChild(QTabWidget, "tabWidget")
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)

        # the new tab button is actually a tab itself, i just make it so that it behaves like a new tab button
        self.tabs.addTab(QWidget(), "")
        self.last_tab_index = self.tabs.count() - 1
        self.tabs.tabBar().setTabButton(self.last_tab_index, QTabBar.RightSide, None)
        self.tabs.tabBar().setTabData(self.last_tab_index, "add_tab_button")
        self.tabs.tabBar().setTabText(self.last_tab_index, "")
        self.tabs.tabBar().tabBarClicked.connect(self.new_tab_button)
        
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
        self.setting_button.clicked.connect(self.on_setting_clicked)
        
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
        self.new_gesture_button.clicked.connect(self.new_gesture_dialog)
        
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
        
        self.BASE_DIR = Path(__file__).parent
        self.PARENT_DIR = self.BASE_DIR.parent
        
        json_files = self.PARENT_DIR.glob("profile_*.json")
        index = 1
        profiles = ProfileManager()
        for files in json_files:
            name = files.stem.replace("profile_", "", 1)
            self.new_tab_button(index)
            self.tabs.setTabText(index, name)
            tab = self.tabs.widget(index)
            scroll = tab.findChild(QScrollArea)
            if scroll.property("individual_sub_bar_container") is True:
                print("Scroll")
                current_scroll_content = scroll.widget()
                current_scroll_layout = current_scroll_content.layout()
                current_profile = profiles.loadProfile(name)
                if current_profile is None:
                    print("[UI] Profile not found / failed to load.")
                    return

                current_action_list = current_profile.getActionList() or []
                
                for act in current_action_list:
            # act is an Actions object
                    g_name = act.getGName()       # gesture name
                    key    = act.getKeyPressed()
                    itype  = act.getInputType()   # "Click" / "Hold" / "D_Click" etc.
                    a_name = act.getName()        # action name (if you want to show it)

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

                    # ---- Gesture Name (use real gesture) ----
                    gesture_name = QTextEdit(g_name, sub_bar_frame)
                    gesture_name.setGeometry(100, 45, 180, 60)
                    gesture_name.setAlignment(Qt.AlignCenter)
                    gFont = gesture_name.font()
                    gFont.setPointSize(14)
                    gesture_name.setFont(gFont)
                    gesture_name.setReadOnly(True)
                    gesture_name.setStyleSheet("""
                        border: none;
                        background: transparent;
                        color: rgb(224, 221, 229);
                    """)

                    # ---- Key Input label ----
                    key_input = QTextEdit("KEY INPUT", sub_bar_frame)
                    key_input.setGeometry(350, 25, 120, 30)
                    key_input.setAlignment(Qt.AlignCenter)
                    kFont = key_input.font()
                    kFont.setPointSize(9)
                    key_input.setFont(kFont)
                    key_input.setReadOnly(True)
                    key_input.setStyleSheet("""
                        border: none;
                        color: rgb(224, 221, 229);
                        background: transparent;
                    """)

                    # ---- Key Input box (show actual key) ----
                    key_input_box = QTextEdit(str(key), sub_bar_frame)
                    key_input_box.setGeometry(325, 55, 150, 40)
                    key_input_box.setAlignment(Qt.AlignCenter)
                    key_input_box.setReadOnly(True)  # set False if you want edit
                    key_input_box.setStyleSheet("""
                        background-color: rgb(224, 221, 229);
                        color: rgb(0, 0, 0);
                    """)

                    # ---- Input Type label ----
                    input_type = QTextEdit("INPUT TYPE", sub_bar_frame)
                    input_type.setGeometry(650, 25, 120, 30)
                    input_type.setAlignment(Qt.AlignCenter)
                    iFont = input_type.font()
                    iFont.setPointSize(9)
                    input_type.setFont(iFont)
                    input_type.setReadOnly(True)
                    input_type.setStyleSheet("""
                        border: none;
                        color: rgb(224, 221, 229);
                        background: transparent;
                    """)

                    # ---- Input Type combo ----
                    input_type_box = QComboBox(sub_bar_frame)
                    input_type_box.setGeometry(625, 55, 150, 40)
                    iFont2 = input_type_box.font()
                    iFont2.setPointSize(9)
                    input_type_box.setFont(iFont2)
                    input_type_box.setStyleSheet("""
                        background-color: rgb(224, 221, 229);
                        color: rgb(0, 0, 0);
                    """)
                    input_type_box.addItems(["Click", "Hold", "Double Click"])
                    for i in range(input_type_box.count()):
                        input_type_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

                    # map profile value to combo display
                    # your backend uses "D_Click" sometimes
                    itype_norm = (itype or "").strip()
                    if itype_norm == "D_Click":
                        itype_norm = "Double Click"
                    # set current selection if match
                    idx = input_type_box.findText(itype_norm)
                    if idx >= 0:
                        input_type_box.setCurrentIndex(idx)

                    # ---- Divider line ----
                    line = QFrame(sub_bar_frame)
                    line.setFrameShape(QFrame.VLine)
                    line.setFrameShadow(QFrame.Sunken)
                    line.setLineWidth(1)
                    line.setGeometry(937, 20, 3, 80)
                    line.setStyleSheet("background-color: rgb(224, 221, 229);")

                    # ---- Action label (DON'T overwrite act variable) ----
                    action_label = QTextEdit("ACTION", sub_bar_frame)
                    action_label.setGeometry(1125, 25, 150, 30)
                    action_label.setAlignment(Qt.AlignCenter)
                    aFont = action_label.font()
                    aFont.setPointSize(9)
                    action_label.setFont(aFont)
                    action_label.setReadOnly(True)
                    action_label.setStyleSheet("""
                        border: none;
                        color: rgb(224, 221, 229);
                        background: transparent;
                    """)

                    # ---- Action box (if you want to show action name) ----
                    action_box = QComboBox(sub_bar_frame)
                    action_box.setGeometry(1100, 55, 200, 40)
                    action_box.setStyleSheet("""
                        background-color: rgb(224, 221, 229);
                        color: rgb(0, 0, 0);
                    """)
                    # If you have a list of possible actions, add them here.
                    # For now just show current action name:
                    action_box.addItem(a_name)

                    # ---- Trash button ----
                    trash_button = QPushButton(sub_bar_widget)
                    trash_button.setGeometry(1450, 20, 80, 80)
                    trash_button.setIcon(QIcon("FYP_GController-main/ML_FILES/designerapp/resource/Recycle-Bin-2--Streamline-Core.png"))
                    trash_button.setIconSize(QSize(50, 50))
                    trash_button.setFlat(True)
                    trash_button.setStyleSheet("""
                        QPushButton:hover {
                            background-color: rgba(255, 255, 255, 0.08);
                            border-radius: 6px
                        }
                    """)
                
                
            index += 1


    # main command sender
    def send_cmd(self, cmd: str):
        try:
            with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=1.0) as s:
                s.settimeout(1.0)
                s.sendall((cmd.strip() + "\n").encode("utf-8"))

                data = s.recv(1024)
                if not data:
                    return "ERR: no response"
                return data.decode("utf-8", errors="ignore").strip()

        except Exception as e:
            return f"ERR: {e}"
        
    #function for the new tab button
    def new_tab_button(self, index):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        library_path = os.path.join(BASE_DIR, "resource", "Definition-Search-Book--Streamline-Core.png")
        plus_path = os.path.join(BASE_DIR, "resource", "Add-Circle--Streamline-Core.png")
        camera_path = os.path.join(BASE_DIR, "resource", "Camera-1--Streamline-Core.png")
        setting_path = os.path.join(BASE_DIR, "resource", "Cog--Streamline-Core.png")
        #check if the tab clicked is the last tab
        if self.tabs.tabBar().tabData(index) == "add_tab_button":
            
            #needed for resize purpose
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
            
            #below are the four buttons inserted to the new tab
            new_four_buttons_container = QWidget(new_tab)
            new_four_buttons_container.setGeometry(1178, 40, 470, 80)
            new_four_buttons_container.move(max(geom.width() - new_four_buttons_container.width()-50, 0), new_four_buttons_container.y())
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
            camera_button.clicked.connect(self.on_camera_clicked)
            camera_button.setGeometry(260, 0, 80,80)
            camera_button.setIcon(QIcon(camera_path))
            camera_button.setIconSize(QSize(50,50))
            camera_button.setFlat(True)
            
            setting_button = QPushButton(new_four_buttons_container)
            setting_button.clicked.connect(self.on_setting_clicked)
            setting_button.setGeometry(390, 0, 80, 80)
            setting_button.setIcon(QIcon(setting_path))
            setting_button.setIconSize(QSize(50,50))
            setting_button.setFlat(True)
            
            #add scroll area to new tab
            scroll_container = QScrollArea(new_tab)
            scroll_container.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll_container.setGeometry(99,150,1599, 962)
            scroll_container.setGeometry(max(x,0), scroll_container.y(), max(geom.width()-scroll_container.x(),0), max(962+geom.height()-1183, 0))
            scroll_content = QWidget()
            scroll_container.setWidget(scroll_content)
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setAlignment(Qt.AlignTop)
            scroll_container.setWidgetResizable(True)
            scroll_container.setStyleSheet ("""
            border: none;                                
                                            """)
            scroll_container.setProperty("individual_sub_bar_container", True)
            
            
            self.tabs.insertTab(insert_index, new_tab, f"Tab {insert_index + 1}")

            # Move "+" tab to the end again
            self.last_tab_index += 1
            self.tabs.tabBar().setTabData(self.last_tab_index, "add_tab_button")
            self.tabs.tabBar().setTabText(self.last_tab_index, "")
            self.tabs.tabBar().setTabButton(self.last_tab_index, QTabBar.RightSide, None)

            # Switch to the newly created tab
            self.tabs.setCurrentIndex(insert_index)

    def close_tab(self, index):
        # Never close the first tab
        if index == 0:
            return

        # Never close the "+" tab
        if self.tabs.tabBar().tabData(index) == "add_tab_button":
            return

        current = self.tabs.currentIndex()

        # Decide where to go AFTER closing (only if user closed the active tab)
        next_index = None
        if index == current:
            # Prefer the tab to the left
            next_index = index - 1

            # Safety: don't land on "+" tab
            if self.tabs.tabBar().tabData(next_index) == "add_tab_button":
                next_index = max(0, next_index - 1)

        # Remove the tab
        self.tabs.removeTab(index)

        # Update "+" tab index (it should always be the last)
        self.last_tab_index = self.tabs.count() - 1

        # Switch to the decided tab
        if next_index is not None:
            # After removal, indices shift if you removed a tab before next_index
            if index < next_index:
                next_index -= 1
            self.tabs.setCurrentIndex(max(0, min(next_index, self.tabs.count() - 2)))

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
                
    def on_library_clicked(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        rec_path   = os.path.join(BASE_DIR, "resource", "Webcam-Video-Circle--Streamline-Core.png")
        trash_path = os.path.join(BASE_DIR, "resource", "Recycle-Bin-2--Streamline-Core.png")

        # ---- locate GestureList.json ----
        # If designerapp is under ML_FILES/designerapp, and GestureList.json is under ML_FILES/
        ML_FILES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # go up from designerapp
        gesturelist_path = os.path.join(ML_FILES_DIR, "GestureList.json")

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
        dialog.setWindowTitle("Gesture Library")
        dialog.setFixedSize(400, 500)
        dialog.setModal(True)

        top_frame = QFrame(dialog)
        top_frame.setGeometry(0, 0, 400, 70)
        top_frame.setStyleSheet("background-color: #030013;")

        label_title = QLabel("Gesture Library", top_frame)
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
                label_title.setText(f"Gesture Library  ({name})")
            else:
                label_title.setText("Gesture Library")

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
                trash_button.setIconSize(QSize(40, 40))
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
        button = self.sender()
        
        # ---- this is to locate the correct scrollarea in each tab
        current_container = button.parent()
        current_tab = current_container.parent()
        current_scroll = current_tab.findChild(QScrollArea)
        current_scroll_content = current_scroll.widget()
        current_scroll_layout = current_scroll_content.layout()
        # Clear existing rows so you don’t duplicate every time you click "+"
        self._clear_layout(current_scroll_layout)

        # Load profile actions
        

        profiles = ProfileManager()
        current_profile = profiles.loadProfile("1")
        if current_profile is None:
            print("[UI] Profile 1 not found / failed to load.")
            return

        current_action_list = current_profile.getActionList() or []

        for act in current_action_list:
            # act is an Actions object
            g_name = act.getGName()       # gesture name
            key    = act.getKeyPressed()
            itype  = act.getInputType()   # "Click" / "Hold" / "D_Click" etc.
            a_name = act.getName()        # action name (if you want to show it)

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

            # ---- Gesture Name (use real gesture) ----
            gesture_name = QTextEdit(g_name, sub_bar_frame)
            gesture_name.setGeometry(100, 45, 180, 60)
            gesture_name.setAlignment(Qt.AlignCenter)
            gFont = gesture_name.font()
            gFont.setPointSize(14)
            gesture_name.setFont(gFont)
            gesture_name.setReadOnly(True)
            gesture_name.setStyleSheet("""
                border: none;
                background: transparent;
                color: rgb(224, 221, 229);
            """)

            # ---- Key Input label ----
            key_input = QTextEdit("KEY INPUT", sub_bar_frame)
            key_input.setGeometry(350, 25, 120, 30)
            key_input.setAlignment(Qt.AlignCenter)
            kFont = key_input.font()
            kFont.setPointSize(9)
            key_input.setFont(kFont)
            key_input.setReadOnly(True)
            key_input.setStyleSheet("""
                border: none;
                color: rgb(224, 221, 229);
                background: transparent;
            """)

            # ---- Key Input box (show actual key) ----
            key_input_box = QTextEdit(str(key), sub_bar_frame)
            key_input_box.setGeometry(325, 55, 150, 40)
            key_input_box.setAlignment(Qt.AlignCenter)
            key_input_box.setReadOnly(True)  # set False if you want edit
            key_input_box.setStyleSheet("""
                background-color: rgb(224, 221, 229);
                color: rgb(0, 0, 0);
            """)

            # ---- Input Type label ----
            input_type = QTextEdit("INPUT TYPE", sub_bar_frame)
            input_type.setGeometry(650, 25, 120, 30)
            input_type.setAlignment(Qt.AlignCenter)
            iFont = input_type.font()
            iFont.setPointSize(9)
            input_type.setFont(iFont)
            input_type.setReadOnly(True)
            input_type.setStyleSheet("""
                border: none;
                color: rgb(224, 221, 229);
                background: transparent;
            """)

            # ---- Input Type combo ----
            input_type_box = QComboBox(sub_bar_frame)
            input_type_box.setGeometry(625, 55, 150, 40)
            iFont2 = input_type_box.font()
            iFont2.setPointSize(9)
            input_type_box.setFont(iFont2)
            input_type_box.setStyleSheet("""
                background-color: rgb(224, 221, 229);
                color: rgb(0, 0, 0);
            """)
            input_type_box.addItems(["Click", "Hold", "Double Click"])
            for i in range(input_type_box.count()):
                input_type_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

            # map profile value to combo display
            # your backend uses "D_Click" sometimes
            itype_norm = (itype or "").strip()
            if itype_norm == "D_Click":
                itype_norm = "Double Click"
            # set current selection if match
            idx = input_type_box.findText(itype_norm)
            if idx >= 0:
                input_type_box.setCurrentIndex(idx)

            # ---- Divider line ----
            line = QFrame(sub_bar_frame)
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setLineWidth(1)
            line.setGeometry(937, 20, 3, 80)
            line.setStyleSheet("background-color: rgb(224, 221, 229);")

            # ---- Action label (DON'T overwrite act variable) ----
            action_label = QTextEdit("ACTION", sub_bar_frame)
            action_label.setGeometry(1125, 25, 150, 30)
            action_label.setAlignment(Qt.AlignCenter)
            aFont = action_label.font()
            aFont.setPointSize(9)
            action_label.setFont(aFont)
            action_label.setReadOnly(True)
            action_label.setStyleSheet("""
                border: none;
                color: rgb(224, 221, 229);
                background: transparent;
            """)

            # ---- Action box (if you want to show action name) ----
            action_box = QComboBox(sub_bar_frame)
            action_box.setGeometry(1100, 55, 200, 40)
            action_box.setStyleSheet("""
                background-color: rgb(224, 221, 229);
                color: rgb(0, 0, 0);
            """)
            # If you have a list of possible actions, add them here.
            # For now just show current action name:
            action_box.addItem(a_name)

            # ---- Trash button ----
            trash_button = QPushButton(sub_bar_widget)
            trash_button.setGeometry(1450, 20, 80, 80)
            trash_button.setIcon(QIcon("FYP_GController-main/ML_FILES/designerapp/resource/Recycle-Bin-2--Streamline-Core.png"))
            trash_button.setIconSize(QSize(50, 50))
            trash_button.setFlat(True)
            trash_button.setStyleSheet("""
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.08);
                    border-radius: 6px
                }
            """)
    
    def new_gesture_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Gesture")
        dialog.setFixedSize(800, 220)
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        title = QTextEdit("Gesture Name")
        title.setAlignment(Qt.AlignCenter)
        title.setReadOnly(True)
        font = title.font()
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        gesture_name_box = QTextEdit("")
        gesture_name_box.setAlignment(Qt.AlignCenter)
        font2 = gesture_name_box.font()
        font2.setPointSize(12)
        gesture_name_box.setFont(font2)
        gesture_name_box.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)
        layout.addWidget(gesture_name_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_ok():
            gname = gesture_name_box.toPlainText().strip()
            if not gname:
                QMessageBox.warning(self, "Error", "Gesture name cannot be empty.")
                return

            resp = self.send_cmd(f"CREATE_GESTURE {gname}")
            if resp.startswith("OK"):
                dialog.accept()
            else:
                QMessageBox.critical(self, "Create Gesture Failed", resp)

        buttons.accepted.connect(on_ok)
        buttons.rejected.connect(dialog.reject)

        dialog.exec()

    
    #for deleting new gesture
    def trash_button_function(self):
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

    def on_camera_clicked(self):
        print(self.send_cmd("TOGGLE_CAMERA"))

    def on_setting_clicked(self):
        print(self.send_cmd("TOGGLE_GUI"))
        

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
            else:
                # letters/numbers/symbols
                key_name = text.lower().strip() if text else f"key_{key}"

            self.setText(key_name)
            self.capturing = False
            return

        super().keyPressEvent(event)

def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    run()
