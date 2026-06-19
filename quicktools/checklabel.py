import cv2
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
IMG_PATH = "Negative/images/-_jpg.rf.c68df3d5f6f7acf0e312b8e7f2711e7b.jpg"
LABEL_PATH = "Negative/negative_labels/-_jpg.rf.c68df3d5f6f7acf0e312b8e7f2711e7b.txt"
IMG_SIZE = 640   # same as YOLO training size

# ----------------------------
# LETTERBOX FUNCTION (YOLO-style)
# ----------------------------
def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]

    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - nw
    pad_h = new_size - nh

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded, scale, left, top


# ----------------------------
# LOAD IMAGE
# ----------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("Could not load image")

orig_h, orig_w = img.shape[:2]

# Apply letterbox (this is what YOLO actually trains on)
img_lb, scale, pad_x, pad_y = letterbox(img, IMG_SIZE)

# ----------------------------
# LOAD & DRAW LABELS
# ----------------------------
with open(LABEL_PATH, "r") as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())

        # Convert YOLO normalized coords → original pixel coords
        x1 = (xc - bw / 2) * orig_w
        y1 = (yc - bh / 2) * orig_h
        x2 = (xc + bw / 2) * orig_w
        y2 = (yc + bh / 2) * orig_h

        # Apply same scale + padding as YOLO
        x1 = int(x1 * scale + pad_x)
        y1 = int(y1 * scale + pad_y)
        x2 = int(x2 * scale + pad_x)
        y2 = int(y2 * scale + pad_y)

        # Draw box
        cv2.rectangle(img_lb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_lb,
            f"class {int(cls)}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# ----------------------------
# SHOW RESULT
# ----------------------------
cv2.imshow("YOLOv6 Letterbox View (What Model Sees)", img_lb)
cv2.waitKey(0)
cv2.destroyAllWindows()