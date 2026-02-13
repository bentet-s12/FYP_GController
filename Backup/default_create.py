import os
import random
import cv2
import numpy as np
import mediapipe as mp


# =========================
#  HAND LANDMARK VECTORIZER
# =========================

class HandLandmarkVectorizer:
    """
    Converts MediaPipe hand landmarks into normalized 42-D vectors:
      - 21 landmarks (x, y)
      - wrist-centered
      - scale-normalized
      - can also return a horizontally mirrored version (x -> -x)
    """

    @staticmethod
    def hand_landmarks_to_vec_pair(hand_lms):
        """
        Convert MediaPipe hand landmarks to TWO normalized 42-D vectors:
          - vec_orig: original orientation
          - vec_mirror: horizontally mirrored (x -> -x)

        Steps:
          - Take 21 landmarks (x, y).
          - Translate so wrist (landmark 0) is at (0,0).
          - Scale so the furthest landmark from the wrist has radius 1.
          - Concatenate xs then ys => 42-D vector.
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


# =========================
#  DATASET BUILDER (OOP)
# =========================

class LandmarkDatasetBuilder:
    """
    Builds a landmark-based gesture dataset from folders of images.

    For each gesture folder:
      - sample up to N images
      - run MediaPipe hand detection
      - extract 42-D landmark vectors (orig + optional mirror)
      - store X, y, class_names to OUT_DIR
      - print stats
    """

    def __init__(
        self,
        src_dir,
        out_dir,
        default_samples=200,
        samples_per_class=None,
        augment_mirror=True,
        image_exts=None,
    ):
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.default_samples = default_samples
        self.samples_per_class = samples_per_class or {}
        self.augment_mirror = augment_mirror
        self.image_exts = image_exts or {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        # To be filled after build()
        self.X = None
        self.y = None
        self.class_names = None
        self.stats = {}  # gesture_name -> dict of counts

    # ---------- helpers ----------

    def _get_image_files(self, folder):
        """Return a list of image file paths in a folder."""
        return [
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if os.path.splitext(fn)[1].lower() in self.image_exts
        ]

    def _detect_gesture_folders(self):
        """Detect subfolders in SRC_DIR that represent gesture classes."""
        if not os.path.isdir(self.src_dir):
            raise FileNotFoundError(f"SRC_DIR does not exist: {self.src_dir}")

        gesture_folders = sorted(
            d for d in os.listdir(self.src_dir)
            if os.path.isdir(os.path.join(self.src_dir, d))
        )
        return gesture_folders

    # ---------- main build logic ----------

    def build(self):
        """
        Run the full pipeline:
          - scan gesture folders
          - extract landmarks
          - save X.npy, y.npy, class_names.npy under out_dir
        """
        print("\n=== BUILDING LANDMARK DATASET FROM IMAGES (OOP) ===")

        os.makedirs(self.out_dir, exist_ok=True)

        gesture_folders = self._detect_gesture_folders()
        if not gesture_folders:
            print(f"[ERROR] No gesture folders found in SRC_DIR: {self.src_dir}")
            return

        print("Detected gesture classes:", gesture_folders)

        X_list = []
        y_list = []
        class_names = []
        stats = {}

        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            for class_id, gesture_name in enumerate(gesture_folders):
                class_names.append(gesture_name)
                stats[gesture_name] = self._process_single_gesture(
                    gesture_name=gesture_name,
                    class_id=class_id,
                    hands=hands,
                    X_list=X_list,
                    y_list=y_list,
                )

        if not X_list:
            print("\n[ERROR] No landmark data collected. "
                  "Check your images / lighting / dataset path.")
            return

        self.X = np.stack(X_list, axis=0)         # (N, 42)
        self.y = np.array(y_list, dtype=np.int64) # (N,)
        self.class_names = np.array(class_names, dtype=object)
        self.stats = stats

        print("\nFinal dataset:")
        print("  X:", self.X.shape)
        print("  y:", self.y.shape)
        print("  classes:", self.class_names)

        self._save()
        self._print_summary()

    def _process_single_gesture(self, gesture_name, class_id, hands, X_list, y_list):
        """
        Process one gesture folder:
          - sample images
          - detect hands
          - extract orig + mirror vectors
        Returns stats dict for this gesture.
        """
        from_dir = os.path.join(self.src_dir, gesture_name)
        img_files = self._get_image_files(from_dir)
        num_found = len(img_files)

        num_requested = self.samples_per_class.get(
            gesture_name, self.default_samples
        )

        if num_found == 0:
            print(f"[WARNING] {gesture_name}: no images found in {from_dir}")
            return {
                "found": 0,
                "requested": num_requested,
                "sampled": 0,
                "detected": 0,
                "vectors": 0,
            }

        # Sample subset
        if num_found <= num_requested:
            sample_files = img_files
            print(
                f"[INFO] {gesture_name}: Using ALL {num_found} images "
                f"(requested {num_requested})"
            )
        else:
            sample_files = random.sample(img_files, num_requested)
            print(
                f"[INFO] {gesture_name}: Using {num_requested}/{num_found} images"
            )

        detected = 0
        vectors_added = 0

        for src_path in sample_files:
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Could not read image: {src_path}")
                continue

            # Ensure 3-channel BGR for MediaPipe
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # BGRA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if not result.multi_hand_landmarks:
                continue

            hand_lms = result.multi_hand_landmarks[0]
            vec_orig, vec_mirror = HandLandmarkVectorizer.hand_landmarks_to_vec_pair(
                hand_lms
            )

            # Always add original vector
            X_list.append(vec_orig)
            y_list.append(class_id)
            vectors_added += 1

            # Optionally add mirrored vector
            if self.augment_mirror:
                X_list.append(vec_mirror)
                y_list.append(class_id)
                vectors_added += 1

            detected += 1

        print(
            f"[DONE] {gesture_name}: "
            f"{detected}/{len(sample_files)} sampled images had a detected hand, "
            f"{vectors_added} landmark vectors added."
        )

        return {
            "found": num_found,
            "requested": num_requested,
            "sampled": len(sample_files),
            "detected": detected,
            "vectors": vectors_added,
        }

    def _save(self):
        """Save X, y, class_names into OUT_DIR."""
        X_out = os.path.join(self.out_dir, "X.npy")
        y_out = os.path.join(self.out_dir, "y.npy")
        classes_out = os.path.join(self.out_dir, "class_names.npy")

        np.save(X_out, self.X)
        np.save(y_out, self.y)
        np.save(classes_out, self.class_names)

        print("\nSaved landmark dataset to:")
        print(" ", X_out)
        print(" ", y_out)
        print(" ", classes_out)

    def _print_summary(self):
        """Print per-gesture statistics as a table."""
        print("\n=== SUMMARY PER GESTURE ===")
        header = f"{'Gesture':<15}{'Found':>8}{'Req':>6}{'Sampled':>10}{'Detected':>10}{'Vectors':>10}"
        print(header)
        print("-" * len(header))

        for gesture_name in self.class_names:
            s = self.stats.get(gesture_name, {})
            print(
                f"{gesture_name:<15}"
                f"{s.get('found', 0):>8}"
                f"{s.get('requested', 0):>6}"
                f"{s.get('sampled', 0):>10}"
                f"{s.get('detected', 0):>10}"
                f"{s.get('vectors', 0):>10}"
            )

        print("\nAll done!\n")


# =========================
#  MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
    # Directory where this script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Source dataset directory: each subfolder is a gesture name
    # e.g. data/datasetDefault/default, data/datasetDefault/point, ...
    SRC_DIR = os.path.join(SCRIPT_DIR, "data", "defaultDataset")

    # Output directory for landmark dataset (X.npy, y.npy, class_names.npy)
    OUT_DIR = os.path.join(SCRIPT_DIR, "data", "landmarkVectors")

    # Default / per-class sample counts
    DEFAULT_SAMPLES = 200
    SAMPLES_PER_CLASS = {
        "default": 400,
        "hold": 200,
        "point": 200,
        "left_click": 200,
    }

    builder = LandmarkDatasetBuilder(
        src_dir=SRC_DIR,
        out_dir=OUT_DIR,
        default_samples=DEFAULT_SAMPLES,
        samples_per_class=SAMPLES_PER_CLASS,
        augment_mirror=True,
    )

    builder.build()
