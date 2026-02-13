import importlib
import os
import subprocess
import sys
from pathlib import Path
from designerapp.D_UI import run
import shutil

# =========================
# REQUIRED DEPENDENCIES
# =========================
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

    raise RuntimeError(
        "Python not found. Install Python 3.10 or 3.11 and ensure it is in PATH."
    )

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

def run_app():
    # Add src to sys.path
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    
    run()

if __name__ == "__main__":
    py = find_python()
    ensure_dependencies(py)
    run_app()
