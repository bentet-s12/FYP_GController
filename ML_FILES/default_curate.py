import os
import random
import shutil

# Base directory of the ASL dataset (adjust this to your directory)
BASE_DIR = r"C:\Users\Nedlanox\Desktop\UOW\FYP\Machine Learning\archive3\American Sign Language Digits Dataset"

# Destination (curated) root
DEST_ROOT = os.path.join(os.path.dirname(BASE_DIR), "curated_asl")

# Mapping: digit folder name -> new gesture name
CLASS_MAP = {
    "1": "point",
    "9": "left_click",
    "5": "hold",
    "0": "default"
}

# Number of images per gesture class (adjust this if the program detects one gesture lesser than the other)
SAMPLES_PER_CLASS = {
    "point": 200,
    "left_click": 200,
    "hold": 200,
    "default": 400
}

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def get_image_files(folder):
    """Return a list of image file paths in a folder."""
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(folder, name))
    return files


def main():
    os.makedirs(DEST_ROOT, exist_ok=True)
    print(f"Destination root: {DEST_ROOT}")

    for digit_name, new_class_name in CLASS_MAP.items():

        # How many images for this gesture?
        num_samples = SAMPLES_PER_CLASS.get(new_class_name, 100)

        # Path to the digit folder (e.g. BASE_DIR/1)
        digit_folder = os.path.join(BASE_DIR, digit_name)
        if not os.path.isdir(digit_folder):
            print(f"[WARNING] Digit folder not found: {digit_folder}")
            continue

        # Expected subfolder: "Input Images - Sign <digit>"
        input_folder_name = f"Input Images - Sign {digit_name}"
        src_folder = os.path.join(digit_folder, input_folder_name)

        if not os.path.isdir(src_folder):
            print(f"[WARNING] Input image folder not found: {src_folder}")
            continue

        # Destination folder
        dest_folder = os.path.join(DEST_ROOT, new_class_name)

        # Get all images inside the nested folder
        img_files = get_image_files(src_folder)
        if len(img_files) == 0:
            print(f"[WARNING] No images found in: {src_folder}")
            continue

        # Sampling
        if len(img_files) < num_samples:
            print(
                f"[WARNING] {src_folder} has only {len(img_files)} images, "
                f"but {num_samples} were requested. Using all available images."
            )
            sample_files = img_files
        else:
            sample_files = random.sample(img_files, num_samples)

        # Prepare destination folder (clear if exists)
        if os.path.exists(dest_folder):
            print(f"[INFO] Clearing existing folder: {dest_folder}")
            shutil.rmtree(dest_folder)
        os.makedirs(dest_folder, exist_ok=True)

        print(f"[INFO] Copying {len(sample_files)} images "
              f"(requested {num_samples}) from {src_folder} -> {dest_folder}")

        # Copy & rename
        for idx, src_path in enumerate(sample_files, start=1):
            ext = os.path.splitext(src_path)[1].lower()
            new_name = f"{new_class_name}_{idx:04d}{ext}"
            dest_path = os.path.join(dest_folder, new_name)
            shutil.copy2(src_path, dest_path)

        print(f"[DONE] {new_class_name}: {len(sample_files)} images saved in {dest_folder}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
