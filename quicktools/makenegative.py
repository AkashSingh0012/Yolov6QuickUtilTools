import os
import glob

# -------- CONFIG --------
BASE_DIR = "Negative"
IMAGE_ROOT = os.path.join(BASE_DIR, "images")
LABEL_ROOT = os.path.join(BASE_DIR, "negative_labels")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
# ------------------------


def create_empty_negative_labels():
    # Ensure base image directory exists
    if not os.path.isdir(IMAGE_ROOT):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_ROOT}")

    # 🔐 Ensure negative_labels directory exists
    os.makedirs(LABEL_ROOT, exist_ok=True)

    # Find all image files recursively
    image_files = glob.glob(
        os.path.join(IMAGE_ROOT, "**", "*"),
        recursive=True
    )

    image_files = [
        f for f in image_files
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"Found {len(image_files)} image files")

    for img_path in image_files:
        # Path relative to images/
        rel_path = os.path.relpath(img_path, IMAGE_ROOT)

        # Convert image path to label path
        label_rel_path = os.path.splitext(rel_path)[0] + ".txt"
        label_path = os.path.join(LABEL_ROOT, label_rel_path)

        label_dir = os.path.dirname(label_path)

        # Create subdirectory structure if needed
        os.makedirs(label_dir, exist_ok=True)

        # Create empty label only if it doesn't exist
        if not os.path.exists(label_path):
            open(label_path, "w").close()
            print(f"Created: {label_path}")
        else:
            print(f"Skipped (exists): {label_path}")


if __name__ == "__main__":
    create_empty_negative_labels()
