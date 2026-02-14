import os
import subprocess
import sys
import shutil

REQUIRED = {
    "PySide6": "PySide6",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "sklearn": "scikit-learn",
    "pyautogui": "pyautogui",
    "pydirectinput": "pydirectinput",
    "keyboard": "keyboard",
    "PIL": "Pillow",
    "SCREENINFO": "screeninfo",
}

# --------------------------------------------------
# Paths
# --------------------------------------------------

def get_project_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_main_path(project_root):
    return os.path.join(project_root, "src", "main.py")


# --------------------------------------------------
# Python Detection
# --------------------------------------------------

def find_python():
    for cmd in ["py", "python", "python3"]:
        path = shutil.which(cmd)
        if path:
            return path
    raise RuntimeError("Python 3.10/3.11 not found in PATH.")


# --------------------------------------------------
# Dependency Installation
# --------------------------------------------------

def ensure_dependencies(python_cmd):
    print("[INFO] Checking dependencies...")

    missing = []

    for pip_name in REQUIRED.values():
        try:
            subprocess.check_output(
                [python_cmd, "-m", "pip", "show", pip_name],
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            missing.append(pip_name)

    if not missing:
        print("[OK] All dependencies already installed.")
        return

    print("[INFO] Installing:", missing)

    subprocess.check_call([
        python_cmd, "-m", "pip", "install", "--upgrade", *missing
    ])


# --------------------------------------------------
# Application Launch
# --------------------------------------------------

def launch_app(python_cmd, main_path):
    print("[INFO] Launching application...")
    subprocess.run([python_cmd, main_path])


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
def main():
    project_root = get_project_root()
    main_path = get_main_path(project_root)

    # CHANGE THIS: Don't use sys.executable (the EXE)
    # Use your find_python() function instead!
    python_cmd = find_python() 

    ensure_dependencies(python_cmd)
    launch_app(python_cmd, main_path)

if __name__ == "__main__":
    main()
