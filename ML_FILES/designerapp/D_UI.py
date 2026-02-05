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
    QSizePolicy, QFrame, QTextBrowser, QGraphicsDropShadowEffect, QTabWidget, QTextEdit, QDialogButtonBox, QInputDialog
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
        
        
        self.profiles = ProfileManager().readFile("profileManager.json")
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
        
        self.tabs.tabBarDoubleClicked.connect(self.tab_rename)
        
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
        RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        json_files = self.PARENT_DIR.glob("profile_*.json")
        index = 1
        
        default_file = ("Default.json")
        default_path = self.PARENT_DIR / default_file
        if (default_path.exists()):
            print ("T")
            default_profile = self.profiles.loadProfile("Default")
            action_list = default_profile.getActionList() or []
            for act in action_list:
                self.build_action_row(self.scroll_layout, profile_id= "Default", act=act)
            else:
                print ("F")
                file_name = "Default"
                path = self.PARENT_DIR / file_name
                try:
                    with open(path, "w") as f:
                        json.dump(self._gestures, f, indent=4)
                except Exception as e:
                    print(f"[Error] Failed to save gestures: {e}") 
        
        for files in json_files:
            name = files.stem.replace("profile_", "", 1)
            self.new_tab_button(index)
            self.tabs.setTabText(index, name)
            tab = self.tabs.widget(index)
            scroll = tab.findChild(QScrollArea)
            if scroll.property("individual_sub_bar_container") is True:
                current_scroll_content = scroll.widget()
                current_scroll_layout = current_scroll_content.layout()
                current_profile = self.profiles.loadProfile(name)
                if current_profile is None:
                    print("[UI] Profile not found / failed to load.")
                    return

                current_action_list = current_profile.getActionList() or []
                
                for act in current_action_list:
                    self.build_action_row(current_scroll_layout, profile_id=name, act=act)
                
            index += 1

        # --- GestureList.json watcher (same polling refresh logic as library dialog) ---
        self._gesturelist_refreshing = False

        try:
            p = self._gesturelist_path()
            self._gesturelist_last_mtime = os.path.getmtime(p) if os.path.exists(p) else None
        except Exception:
            self._gesturelist_last_mtime = None

        self._gesturelist_timer = QTimer(self)
        self._gesturelist_timer.timeout.connect(self._tick_gesturelist_refresh)
        self._gesturelist_timer.start(500)




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
        
    def tab_rename(self, index):
        text, ok = QInputDialog.getText(
        self,
        "Rename Tab",
        "Enter new name:"
        )
        print (self.tabs.tabText(index))
        if ok and text:
            current_name = self.tabs.tabText(index)
            print (current_name)
            self.profiles.renameProfile(current_name, text)
            self.tabs.setTabText(index, text)
    
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
            self._refresh_action_dropdowns_from_gesturelist()
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
        # your profiles live in ML_FILES (parent of designerapp)
        return str(self.PARENT_DIR / f"profile_{profile_id}.json")

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

        base_dir = os.path.dirname(os.path.abspath(__import__("ProfileManager").__file__))
        profile_path = os.path.join(base_dir, f"profile_{profile_id}.json")

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
        # GestureList.json is in ML_FILES (parent of designerapp)
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

        # ----- Editable NAME field (this edits "name") -----
        gesture_edit = QLineEdit(a_name, sub_bar_frame)
        gesture_edit.setGeometry(100, 55, 180, 40)
        gesture_edit.setAlignment(Qt.AlignCenter)
        gFont = gesture_edit.font()
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
        kFont.setPointSize(9)
        key_input.setFont(kFont)
        key_input.setReadOnly(True)
        key_input.setStyleSheet("""
            border: none;
            color: rgb(224, 221, 229);
            background: transparent;
        """)

        # ----- Editable KEY field (edits "key_pressed") -----
        key_edit = QLineEdit("" if key is None else str(key), sub_bar_frame)
        key_edit.setGeometry(325, 55, 150, 40)
        key_edit.setAlignment(Qt.AlignCenter)
        key_edit.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)

        # ----- Input type label -----
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

        # ----- Input type dropdown (edits "input_type") -----
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
        aFont.setPointSize(9)
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
                new_key=key_edit.text().strip(),
                new_input_type=input_type_box.currentText(),
                new_name=new_id                               # writes to name
            )

            # if rename succeeded logically, update our local id so future edits & delete work
            if new_id and new_id != old_id:
                action_id_ref["id"] = new_id


        gesture_edit.editingFinished.connect(commit_change)
        key_edit.editingFinished.connect(commit_change)
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
