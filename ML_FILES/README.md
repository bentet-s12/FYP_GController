Put this in your terminal to install the required libraries:
pip install numpy scikit-learn opencv-python mediapipe pyautogui matplotlib

default_create is where you create the initial default gestures.
If you want to use another default dataset then do this:
In folder '.\data\defaultDataset', replace each gesture in default, hold, left_click and point.
The standard number of images per gesture is 200, but you can change it in default_create.py.

The user can run the program by calling GestureControllerApp class from prototype.py.

In custom_manager, the user can create, edit and delete their custom gestures.
