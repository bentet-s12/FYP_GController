import designerapp.resources_rc
import subprocess
import sys
import os
import threading
import socket
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import QSize
from PySide6.QtWidgets import ( 
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QStatusBar, QMessageBox, QLabel, QPushButton, 
    QLineEdit, QComboBox, QTabBar, QToolButton, QDialog, QScrollArea, 
    QSizePolicy, QFrame, QTextBrowser, QGraphicsDropShadowEffect, QTabWidget
)

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

        
        # got rid of close button for the default/first tab
        default_close_button = self.tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        default_close_button.hide()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        proto_path = os.path.join(project_root, "prototypeV2.py")

        # Start ONE background process (only once)
        self._backend_proc = start_backend_if_needed(
            proto_path=proto_path,
            project_root=project_root,
            port=50555
        )


    # main command sender
    def send_cmd(self, cmd: str):
        try:
            with socket.create_connection(("127.0.0.1", 50555), timeout=0.3) as s:
                s.sendall((cmd + "\n").encode("utf-8"))
                return s.recv(1024).decode("utf-8", errors="ignore").strip()
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
            new_four_buttons_container.setGeometry(1178, 50, 470, 80)
            new_four_buttons_container.move(max(geom.width() - new_four_buttons_container.width()-50, 0), new_four_buttons_container.y())
            new_four_buttons_container.setProperty("4_buttons_container", True)
            
            library_button = QPushButton(new_four_buttons_container)
            library_button.setGeometry(0,0,80,80)
            library_button.setIcon(QIcon(library_path))
            library_button.setIconSize(QSize(50, 50))
            library_button.setFlat(True)
            
            plus_button = QPushButton(new_four_buttons_container)
            plus_button.setGeometry(130,0,80,80)
            plus_button.setIcon(QIcon(plus_path))
            plus_button.setIconSize(QSize(50,50))
            plus_button.setFlat(True)
            
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
            i.move(max(x,0), i.y())
            
        for b in buttons_containers:
            b.move(max(geom.width() - b.width()-50, 0), b.y())
        
        super().resizeEvent(event)

    def on_camera_clicked(self):
        print(self.send_cmd("TOGGLE_CAMERA"))

    def on_setting_clicked(self):
        print(self.send_cmd("TOGGLE_GUI"))

        

def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    run()
