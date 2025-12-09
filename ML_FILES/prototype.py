import os
import time
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import pyautogui
import tkinter as tk
import mediapipe as mp
import ProfileManager as profile_manager
# ========= CONFIG =========

# Folder containing landmark-based X.npy, y.npy, class_names.npy
# adjust this to your directory
BASE_DIR = r"C:\Users\Nedlanox\Desktop\UOW\FYP\Machine Learning\data\default"

K_NEIGHBORS = 3
SMOOTH_WINDOW = 3               # smoothing window for action-hand gesture prediction
GESTURE_CONF_THRESHOLD = 0.6    # min probability to trust a gesture


# ========= LOAD DATA FOR KNN (LANDMARK FEATURES, 42-D) =========

X_path = os.path.join(BASE_DIR, "X.npy")
y_path = os.path.join(BASE_DIR, "y.npy")
classes_path = os.path.join(BASE_DIR, "class_names.npy")

X = np.load(X_path)                         # (N, 42)
y = np.load(y_path)                         # (N,)
class_names = np.load(classes_path, allow_pickle=True)

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)
print("Classes:", class_names)

knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn.fit(X, y)
print("k-NN trained on 42-D landmark vectors.")

id_to_name = {i: name for i, name in enumerate(class_names)}

# ========= MEDIAPIPE HANDS =========

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ========= TKINTER GUI =========
# for debug, comment out if not needed

click_count = 0

root = tk.Tk()
root.title("Gesture Click Tester")
root.geometry("380x260")

click_var = tk.StringVar()
click_var.set("Clicks: 0")


def increment_click_counter():
    global click_count
    click_count += 1
    click_var.set(f"Clicks: {click_count}")


# Big button that shows number of clicks; any real left-click on it increments 
btn = tk.Button(
    root,
    textvariable=click_var,
    font=("Arial", 20),
    width=10,
    height=2,
    command=increment_click_counter,
)
btn.pack(expand=True, fill="both", pady=10)


# ========= HELPERS =========

def hand_landmarks_to_vec_pair(hand_lms):
    """
    Convert MediaPipe hand landmarks to TWO normalized 42-D vectors:
      - vec_orig: original orientation
      - vec_mirror: horizontally mirrored around the wrist (xs -> -xs)
    This makes classification robust to left/right hand mirroring.
    """
    xs = []
    ys = []
    for lm in hand_lms.landmark:
        xs.append(lm.x)
        ys.append(lm.y)

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Translate so wrist is origin
    xs -= xs[0]
    ys -= ys[0]

    # Scale by max distance from wrist
    radii = np.sqrt(xs**2 + ys**2)
    max_r = np.max(radii)
    if max_r > 0:
        xs_norm = xs / max_r
        ys_norm = ys / max_r
    else:
        xs_norm = xs
        ys_norm = ys

    # Original orientation
    vec_orig = np.concatenate([xs_norm, ys_norm], axis=0)  # (42,)

    # Mirrored horizontally (x -> -x)
    xs_mirror = -xs_norm
    vec_mirror = np.concatenate([xs_mirror, ys_norm], axis=0)

    return vec_orig, vec_mirror


def classify_hand_from_landmarks(hand_lms):
    """
    Try both original and mirrored landmark vectors and keep the more confident one.
    Returns (raw_label, confidence).
    """
    vec_orig, vec_mirror = hand_landmarks_to_vec_pair(hand_lms)

    best_label = "none"
    best_conf = -1.0

    for vec in (vec_orig, vec_mirror):
        proba = knn.predict_proba([vec])[0]  # (num_classes,)
        top_idx = int(np.argmax(proba))
        top_conf = float(proba[top_idx])
        if top_conf > best_conf:
            best_conf = top_conf
            best_label = id_to_name[top_idx]

    return best_label, best_conf


def get_hands_info(results, frame_w, frame_h):
    """
    Build list of detected hands with:
      - mp_label: MediaPipe "Left"/"Right" (for debug)
      - landmarks
      - fingertip position
      - center_x: average x over all landmarks (for debug / optional use)
    """
    infos = []
    if not (results.multi_hand_landmarks and results.multi_handedness):
        return infos

    for hand_lms, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        mp_label = handedness.classification[0].label  # "Left" / "Right"

        # fingertip pixel position
        tip = hand_lms.landmark[8]  # index fingertip
        fx = int(tip.x * frame_w)
        fy = int(tip.y * frame_h)

        # center x in image coords (0..frame_w)
        xs = [lm.x * frame_w for lm in hand_lms.landmark]
        center_x = float(np.mean(xs))

        infos.append(
            {
                "hand_lms": hand_lms,
                "mp_label": mp_label,   # for debug only
                "center_x": center_x,
                "tip_px": (fx, fy),
                "tip_norm": (tip.x, tip.y),
                "raw_label": "none",
                "raw_conf": 0.0,
            }
        )

    return infos


# ========= REAL-TIME LOOP =========

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    raise SystemExit

screen_w, screen_h = pyautogui.size()
print("Screen size:", screen_w, "x", screen_h)

last_action_predictions = []   # logical predictions for action hand: "left_click", "hold", or "none"
last_smoothed_action = "none"
is_holding = False             # whether LMB is currently held down

print("Press 'q' in the OpenCV window to quit.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            # Keep Tkinter GUI responsive
            # comment out if not needed
            root.update_idletasks()
            root.update()

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # mirror view
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # detect all hands
            hands_info = get_hands_info(results, w, h)

            # classify each hand
            for info in hands_info:
                raw_label, raw_conf = classify_hand_from_landmarks(info["hand_lms"])
                info["raw_label"] = raw_label
                info["raw_conf"] = raw_conf

            # determine pointer hand (gesture == 'point')
            pointer_hand = None
            pointer_debug = "None"
            if hands_info:
                # candidates: hand(s) with label 'point' and enough confidence
                point_candidates = [
                    hinfo for hinfo in hands_info
                    if hinfo["raw_label"] == "point" and hinfo["raw_conf"] >= GESTURE_CONF_THRESHOLD
                ]
                if point_candidates:
                    # choose the one with highest confidence
                    pointer_hand = max(point_candidates, key=lambda hinfo: hinfo["raw_conf"])
                else:
                    # fallback: if no 'point', use the first detected hand as pointer
                    pointer_hand = hands_info[0]

            # move cursor with pointer hand
            if pointer_hand is not None:
                fx, fy = pointer_hand["tip_px"]
                nx, ny = pointer_hand["tip_norm"]

                # draw fingertip for pointer hand
                cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

                sx = int(nx * screen_w)
                sy = int(ny * screen_h)
                pyautogui.moveTo(sx, sy, duration=0)

                pointer_debug = f"{pointer_hand['mp_label']} ({pointer_hand['raw_label']}, {pointer_hand['raw_conf']:.2f})"

            # determine action hand (left_click / hold)
            action_hand = None
            action_raw_label = "none"
            action_raw_conf = 0.0
            action_debug = "None"

            if hands_info:
                # candidates: not the pointer OR can be same if only one hand
                for hinfo in hands_info:
                    # allow same hand to be both pointer and action if only one exists
                    is_same_as_pointer = (hinfo is pointer_hand)
                    if len(hands_info) > 1 and is_same_as_pointer:
                        continue  # skip pointer, use other hand(s) as action

                    if hinfo["raw_conf"] < GESTURE_CONF_THRESHOLD:
                        continue

                    if hinfo["raw_label"] in ("left_click", "hold"):
                        # pick the most confident action gesture
                        if hinfo["raw_conf"] > action_raw_conf:
                            action_raw_conf = hinfo["raw_conf"]
                            action_raw_label = hinfo["raw_label"]
                            action_hand = hinfo

            if action_hand is not None:
                action_debug = f"{action_hand['mp_label']} ({action_raw_label}, {action_raw_conf:.2f})"

                # draw skeleton for debug on action hand
                mp_drawing.draw_landmarks(
                    frame,
                    action_hand["hand_lms"],
                    mp_hands.HAND_CONNECTIONS,
                )
            else:
                action_raw_label = "none"
                action_raw_conf = 0.0

            # ---- MAP RAW LABEL -> LOGICAL LABEL (left_click / hold / none) ----
            if action_raw_label == "left_click":
                logical_label = "left_click"
            elif action_raw_label == "hold":
                logical_label = "hold"
            else:
                logical_label = "none"

            # ---- SMOOTH ACTION HAND PREDICTIONS ----
            last_action_predictions.append(logical_label)
            if len(last_action_predictions) > SMOOTH_WINDOW:
                last_action_predictions.pop(0)

            counts = {
                "left_click": last_action_predictions.count("left_click"),
                "hold": last_action_predictions.count("hold"),
                "none": last_action_predictions.count("none"),
            }
            smoothed_action = max(counts, key=counts.get)

            prev_action = last_smoothed_action
            last_smoothed_action = smoothed_action

            # ========= COMMANDS FROM ACTION HAND =========
            # left_click: single click on transition into left_click
            active_profile = Profile("active")
            active_profile.addAction(Actions("none", "space", "Click"))
            active_profile.addAction(Actions("hold", "w", "Hold"))
            active_profile.addAction(Actions("left_click", "left", "Click"))
            active_profile = profile_manager.getProfile(active_profile.getProfileId())  # get current profile

<<<<<<< HEAD
            if active_profile.getAction(smoothed_action) is not None:
                active_profile.callfunction(smoothed_action)
            else:
                print("Gesture not mapped to any action in this profile.")

            # ---- DEBUG TEXT ----
=======
            # get the action object
            action_obj = active_profile.getAction(smoothed_action)

            if action_obj is not None:
                # --- set holdvalue based on gesture ---
                if smoothed_action == "hold":
                    action_obj.setholdvalue(True)    # start holding
                else:
                    action_obj.setholdvalue(False)   # stop holding if it was held

                # trigger the action
                active_profile.callfunction(smoothed_action)
            else:
                print("Gesture not mapped to any action in this profile.")
                    # ---- DEBUG TEXT ----
>>>>>>> 94a8dc0833c58e0f8155f73cd5114c9dd4c61305
            debug_text = (
                f"Pointer: {pointer_debug} | "
                f"Action: {action_debug} | "
                f"Smoothed action: {smoothed_action}"
            )
            cv2.putText(
                frame,
                debug_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Gesture Pointer Prototype (Seamless Roles)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
    if is_holding:
        pyautogui.mouseUp(button="left")
    try:
        root.destroy()
    except tk.TclError:
        pass
