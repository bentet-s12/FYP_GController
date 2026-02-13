import importlib
import os
import subprocess
import sys
import shutil
from designerapp.D_UI import run
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
}

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
main_path = os.path.join(src_path, "main.py")

def find_python():
    py = shutil.which("py")
    if py:
        return py
    pyexe = shutil.which("python")
    if pyexe:
        return pyexe
    py3 = shutil.which("python3")
    if py3:
        return py3
    raise RuntimeError("Python not found. Install Python 3.10/3.11 and add it to PATH.")

def ensure_dependencies(python_cmd):
    missing = []
    for import_name, pip_name in REQUIRED.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(pip_name)

    if not missing:
        print("[OK] All dependencies installed.")
        return

    print("[INFO] Installing missing dependencies:", missing)
    subprocess.check_call([python_cmd, "-m", "pip", "install", "--upgrade", *missing])

if __name__ == "__main__":
    py = find_python()
    ensure_dependencies(py)
    
    subprocess.run([sys.executable, main_path])
