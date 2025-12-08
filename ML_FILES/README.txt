Put this in your terminal to install the required libraries:
pip install numpy scikit-learn opencv-python mediapipe pyautogui matplotlib

Default gestures:
Run in this order;
default_curate - extracts set amount of images at random from a dataset
default_convertNpy.py - converts the images into landmark vectors for recognition
knn_classifier.py - the main model for this project (output is for debug)
prototype.py - view the gestures that can be used in the default_curate output folder
