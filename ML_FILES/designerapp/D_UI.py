import resources_rc
import keyboard
import subprocess
import sys
import os
import threading
import socket
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QIODevice
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtCore import QSize
from PySide6.QtWidgets import ( 
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QStatusBar, QMessageBox, QLabel, QPushButton, 
    QLineEdit, QComboBox, QTabBar, QToolButton, QDialog, QScrollArea, 
    QSizePolicy, QFrame, QTextBrowser, QGraphicsDropShadowEffect, QTabWidget, QTextEdit, QDialogButtonBox
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
            scroll_container.setGeometry(99,150,1599, 962)
            scroll_container.setGeometry(max(x,0), scroll_container.y(), max(geom.width()-scroll_container.x(),0), max(962+geom.height()-1183, 0))
            scroll_content = scroll_container.widget()
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setAlignment(Qt.AlignTop)
            scroll_container.setWidgetResizable(True)
            scroll_container.setStyleSheet ("""
            border: none;                                
                                            """)
            
            
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


    def new_gesture_button_function(self):
        current_index = self.tabs.currentIndex()
        current_tab_name = self.tabs.tabText(current_index)

        from ProfileManager import ProfileManager
        profiles = ProfileManager()
        current_profile = profiles.loadProfile("1")
        current_action_list = current_profile.getActionList()
        for i in len(current_action_list):
        
            sub_bar_widget = QWidget()
            sub_bar_widget.setFixedHeight(150)
            self.scroll_layout.addWidget(sub_bar_widget)
            sub_bar_frame = QFrame(sub_bar_widget)
            sub_bar_frame.setGeometry(0,0,1400,125)
            sub_bar_frame.setProperty("individual_sub_bar", True)
            sub_bar_frame.setStyleSheet ("""
                QFrame[individual_sub_bar] {
    background-color: #252438;
    border-radius: 12px;
    }                         
                                        """)
                
            gesture_name = QTextEdit("TEST", sub_bar_frame)
            gesture_name.setGeometry(100,45,100,60)
            gesture_name.setAlignment(Qt.AlignCenter)
            gFont = gesture_name.font()
            gFont.setPointSize(14)
            gesture_name.setFont(gFont)
            gesture_name.setReadOnly(True)
            gesture_name.setStyleSheet ("""
                border: none;
    background: transparent;
    color: rgb(224, 221, 229);                       
                                        """)
            
            key_input = QTextEdit("KEY INPUT", sub_bar_frame)
            key_input.setGeometry(350,25,100,30)
            key_input.setAlignment(Qt.AlignCenter)
            kFont = key_input.font()
            kFont.setPointSize(9)
            key_input.setFont(kFont)
            key_input.setReadOnly(True)
            key_input.setStyleSheet ("""
    border: none;
    color: rgb(224, 221, 229);
    background: transparent;                      
                                        """)
            
            key_input_box = QTextEdit("",sub_bar_frame)
            key_input_box.setGeometry(325,55,150,40)
            key_input_box.setAlignment(Qt.AlignCenter)
            kFont2 = key_input_box.font()
            key_input_box.setFont(kFont2)
            key_input_box.setReadOnly(True)
            key_input_box.setStyleSheet ("""
    background-color: rgb(224, 221, 229);
    color: rgb(0, 0, 0);                     
                                        """)
            
            input_type = QTextEdit("INPUT TYPE", sub_bar_frame)
            input_type.setGeometry(650,25,100,30)
            input_type.setAlignment(Qt.AlignCenter)
            iFont = input_type.font()
            iFont.setPointSize(9)
            input_type.setFont(iFont)
            input_type.setReadOnly(True)
            input_type.setStyleSheet ("""
    border: none;
    color: rgb(224, 221, 229);
    background: transparent;                  
                                        """)
            
            input_type_box = QComboBox(sub_bar_frame)
            input_type_box.setGeometry(625, 55, 150, 40)
            iFont2 = input_type_box.font()
            iFont2.setPointSize(9)
            input_type_box.setFont(iFont2)
            input_type_box.setStyleSheet ("""
    background-color: rgb(224, 221, 229);
    color: rgb(0, 0, 0);                  
                                        """)
            input_type_box.addItems ([
                "Click",
                "Hold",
                "Double Click"
            ])
            
            for i in range(input_type_box.count()):
                input_type_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            
            line = QFrame(sub_bar_frame)
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setLineWidth(1)
            line.setGeometry(937,20,3,80)
            line.setStyleSheet ("""
    background-color: rgb(224, 221, 229);                            
                                """)
            
            action = QTextEdit("ACTION", sub_bar_frame)
            action.setGeometry(1125,25,150,30)
            action.setAlignment(Qt.AlignCenter)
            aFont = action.font()
            aFont.setPointSize(9)
            action.setFont(aFont)
            action.setReadOnly(True)
            action.setStyleSheet ("""
    border: none;
    color: rgb(224, 221, 229);
    background: transparent;                  
                                        """)
            
            action_box = QComboBox(sub_bar_frame)
            action_box.setGeometry(1100,55,200,40)
            action_box.setStyleSheet ("""
    background-color: rgb(224, 221, 229);
    color: rgb(0, 0, 0);                  
                                        """)
            
            trash_button = QPushButton(sub_bar_widget)
            trash_button.setGeometry(1450,20,80,80)
            trash_button.setIcon(QIcon("FYP_GController-main/ML_FILES/designerapp/resource/Recycle-Bin-2--Streamline-Core.png"))
            trash_button.setIconSize(QSize(50,50))
            trash_button.setFlat(True)
            trash_button.setStyleSheet ("""
    QPushButton:hover { background-color: rgba(255, 255, 255, 0.08); 
    border-radius: 6px}
                    
                                        """)
    
    #the new gesture button will be connected to this function now, for the user to key in the information for new gesture
    #the ok button in the dialog wll trigger the function to add new gesture        
    def new_gesture_dialog(self):
        dialog = QDialog(self)
        dialog = QDialog(self)
        dialog.setWindowTitle("Modal Dialog")
        dialog.setFixedSize(800, 300)
        dialog.setModal(True)  # Make it modal

        layout = QVBoxLayout()

        gesture_name = QTextEdit("Name of Gesture")
        gesture_name.setAlignment(Qt.AlignCenter)
        gesture_name.setReadOnly(True)
        gFont = gesture_name.font()
        gFont.setPointSize(12)
        gesture_name.setFont(gFont)
        layout.addWidget(gesture_name)

        gesture_name_box = QTextEdit("")
        gesture_name_box.setAlignment(Qt.AlignCenter)
        gFont2 = gesture_name_box.font()
        gFont2.setPointSize(12)
        gesture_name_box.setFont(gFont2)
        gesture_name_box.setStyleSheet ("""
    background-color: rgb(224, 221, 229);
    color: rgb(0, 0, 0);
                    
                                        """)
        layout.addWidget(gesture_name_box)
        
        key_input = QTextEdit("Input Key")
        key_input.setAlignment(Qt.AlignCenter)
        key_input.setReadOnly(True)
        kfont = key_input.font()
        kfont.setPointSize(12)
        key_input.setFont(kfont)
        layout.addWidget(key_input)
        
        key_input_box = KeyCaptureBox()
        layout.addWidget(key_input_box)
        
        
        
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.new_gesture_button_function)  # Close dialog when OK is clicked
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        dialog.exec()  # Blocks interaction with the main window
    
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
        self.setReadOnly(True)  # User cannot type freely
        self.setStyleSheet("""
            background-color: rgb(224, 221, 229);
            color: rgb(0, 0, 0);
        """)
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.capturing = False  # Only capture key after clicked

    def mousePressEvent(self, event):
        # When clicked, start capturing next key
        self.setText("Press a key...")
        self.capturing = True
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self.capturing:
            key_name = keyboard.read_key()  # Fallback for non-character keys
            self.setText(f"{key_name}")
            self.capturing = False  # Stop capturing after one key
        else:
            super().keyPressEvent(event)

        

def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    run()
