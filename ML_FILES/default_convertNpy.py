import os
import cv2
import numpy as np
import mediapipe as mp

# ===== CONFIG =====

# Base folder that contains gesture subfolders:
#   default/, point/, left_click/, hold/
#   adjust this to your directory
BASE_DIR = r"C:\Users\Nedlanox\Desktop\UOW\FYP\Machine Learning\data\default"

# Folder names = class names (must match your gesture folders exactly)
CLASS_NAMES = ["default", "point", "left_click", "hold"]

# If True, for every detected hand we add:
#   - original landmark vector
#   - horizontally mirrored vector (x -> -x)
AUGMENT_MIRROR = True


def hand_landmarks_to_vec_pair(hand_lms):
    """
    Convert MediaPipe hand landmarks to TWO normalized 42-D vectors:
      - vec_orig: original orientation
      - vec_mirror: horizontally mirrored around the wrist (xs -> -xs)
    Steps:
      - Use wrist (landmark 0) as origin.
      - Scale by max distance from wrist (size invariance).
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


def main():
    mp_hands = mp.solutions.hands

    X_list = []
    y_list = []

    total_imgs = 0
    total_detected = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        for class_id, class_name in enumerate(CLASS_NAMES):
            folder = os.path.join(BASE_DIR, class_name)
            if not os.path.isdir(folder):
                print(f"[WARNING] Folder not found for class '{class_name}': {folder}")
                continue

            print(f"\nProcessing class '{class_name}' (label {class_id})")
            file_names = sorted(os.listdir(folder))
            if not file_names:
                print(f"[WARNING] No files in {folder}")
                continue

            class_total = 0
            class_detected = 0

            for fname in file_names:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue

                path = os.path.join(folder, fname)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARNING] Could not read image: {path}")
                    continue

                # Make sure we have 3-channel BGR for MediaPipe
                if len(img.shape) == 2:  # grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # BGRA -> BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                class_total += 1
                total_imgs += 1

                # MediaPipe expects RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if not result.multi_hand_landmarks:
                    # print(f"[WARNING] No hand detected in {path}")
                    continue

                hand_lms = result.multi_hand_landmarks[0]
                vec_orig, vec_mirror = hand_landmarks_to_vec_pair(hand_lms)

                # Always add original
                X_list.append(vec_orig)
                y_list.append(class_id)

                # Optionally add mirrored copy
                if AUGMENT_MIRROR:
                    X_list.append(vec_mirror)
                    y_list.append(class_id)

                class_detected += 1
                total_detected += 1

            print(f"[INFO] Class '{class_name}': {class_detected}/{class_total} images with detected hands.")

    if not X_list:
        print("\n[ERROR] No landmark data collected. "
              "Check your images / lighting / dataset.")
        return

    X = np.stack(X_list, axis=0)            # (N, 42)
    y = np.array(y_list, dtype=np.int64)    # (N,)
    class_names_np = np.array(CLASS_NAMES, dtype=object)

    print("\nFinal dataset:")
    print("  X:", X.shape)
    print("  y:", y.shape)
    print("  classes:", class_names_np)
    print(f"Overall detection rate (per-image): "
          f"{total_detected}/{total_imgs} images with at least one hand.")

    # Save into BASE_DIR (overwrites old npy)
    X_out = os.path.join(BASE_DIR, "X.npy")
    y_out = os.path.join(BASE_DIR, "y.npy")
    classes_out = os.path.join(BASE_DIR, "class_names.npy")

    np.save(X_out, X)
    np.save(y_out, y)
    np.save(classes_out, class_names_np)

    print("\nSaved landmark dataset to:")
    print(" ", X_out)
    print(" ", y_out)
    print(" ", classes_out)


if __name__ == "__main__":
    main()
