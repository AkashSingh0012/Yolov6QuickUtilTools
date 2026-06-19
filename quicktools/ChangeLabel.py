import os

def update_yolov6_labels(label_dir, image_dir, new_class_id):
    """
    Updates ONLY the class ID (first value) in YOLOv6 label files.
    Bounding box values remain unchanged.
    """

    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    if not os.path.isdir(image_dir):
        print(f"[WARNING] Image folder not found: {image_dir}")
        print("Continuing without image validation...\n")

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    if not label_files:
        print("No label files found.")
        return

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        # Optional image existence check
        image_name = os.path.splitext(label_file)[0]
        image_exists = any(
            os.path.exists(os.path.join(image_dir, image_name + ext))
            for ext in [".jpg", ".jpeg", ".png"]
        )

        if not image_exists:
            print(f"[INFO] No image found for {label_file}, still updating label.")

        updated_lines = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(f"[SKIP] Invalid label format in {label_file}: {line}")
                continue

            # Replace ONLY class ID
            parts[0] = str(new_class_id)
            updated_lines.append(" ".join(parts))

        # Write back updated labels
        with open(label_path, "w") as f:
            f.write("\n".join(updated_lines) + "\n")

        print(f"[UPDATED] {label_file}")

    print("\n All labels updated successfully.")


if __name__ == "__main__":
    print("YOLOv6 Label Class Updater\n")

    label_folder = input("Enter path to Labels folder: ").strip()
    image_folder = input("Enter path to Images folder: ").strip()
    new_class = input("Enter NEW class ID (integer): ").strip()

    if not new_class.isdigit():
        raise ValueError("Class ID must be an integer.")

    update_yolov6_labels(
        label_dir=label_folder,
        image_dir=image_folder,
        new_class_id=int(new_class)
    )
