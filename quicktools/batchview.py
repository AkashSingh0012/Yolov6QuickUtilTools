import cv2
import os
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_DIR = Path("D:/detector/YOLOv6/data/images/train")
QUARANTINE_DIR = Path("D:/detector/quicktools/quarantine")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 6
THUMB_SIZE = 320  # size of each thumbnail in batch
X_BTN_SIZE = (25, 20)  # width, height of the "X" button

# ---------------- HELPERS ----------------
def letterbox(img, new_size=THUMB_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = new_size - nw, new_size - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top

def load_labels(label_path, img_shape):
    h, w = img_shape[:2]
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            boxes.append((x1, y1, x2, y2, int(cls)))
    return boxes

def move_to_quarantine(img_path, lbl_path):
    rel_img = img_path.relative_to(DATA_DIR / "images")
    rel_lbl = lbl_path.relative_to(DATA_DIR / "labels")
    dest_img = QUARANTINE_DIR / "images" / rel_img
    dest_lbl = QUARANTINE_DIR / "labels" / rel_lbl
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), str(dest_img))
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(dest_lbl))
    print(f"[QUARANTINED] {img_path.name}")

# ---------------- LOAD DATA ----------------
all_images = sorted([p for p in (DATA_DIR / "images").rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])
index = 0

# ---------------- GUI CALLBACK ----------------
quarantined = set()
current_batch = []

def click_event(event, x, y, flags, param):
    global current_batch, quarantined
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is in X area of any image
        for i, (img_path, top_left) in enumerate(current_batch):
            x0, y0 = top_left
            x1, y1 = x0 + THUMB_SIZE, y0 + THUMB_SIZE
            # X button top-right
            btn_x0 = x1 - X_BTN_SIZE[0]
            btn_y0 = y0
            btn_x1 = x1
            btn_y1 = y0 + X_BTN_SIZE[1]
            if btn_x0 <= x <= btn_x1 and btn_y0 <= y <= btn_y1:
                # Quarantine this image
                lbl_path = DATA_DIR / "labels" / img_path.relative_to(DATA_DIR / "images").with_suffix(".txt")
                move_to_quarantine(img_path, lbl_path)
                quarantined.add(img_path)
                # Draw red overlay to indicate it was quarantined
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0,0,255), 3)
                cv2.imshow("Batch Viewer", canvas)
                break

cv2.namedWindow("Batch Viewer")
cv2.setMouseCallback("Batch Viewer", click_event)

# ---------------- BATCH VIEWER ----------------
while index < len(all_images):
    batch = all_images[index:index+BATCH_SIZE]
    canvas_rows = []
    current_batch = []

    for img_path in batch:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = DATA_DIR / "labels" / img_path.relative_to(DATA_DIR / "images").with_suffix(".txt")
        img_lb, scale, pad_x, pad_y = letterbox(img, THUMB_SIZE)
        boxes = load_labels(lbl_path, img.shape)
        for x1, y1, x2, y2, cls in boxes:
            cv2.rectangle(img_lb, (x1//2, y1//2), (x2//2, y2//2), (0, 255, 0), 1)
            cv2.putText(img_lb, f"{cls}", (x1//2, max(y1//2-3,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        # Draw X button
        cv2.rectangle(img_lb, (THUMB_SIZE-X_BTN_SIZE[0],0), (THUMB_SIZE, X_BTN_SIZE[1]), (0,0,255), -1)
        cv2.putText(img_lb, "X", (THUMB_SIZE-22,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        current_batch.append((img_path, (0,0)))  # placeholder top-left
        canvas_rows.append(img_lb)

    if not canvas_rows:
        break
    # Concatenate horizontally for now
    canvas = cv2.hconcat(canvas_rows)
    # Update top-left positions for each image for mouse click detection
    x_offset = 0
    for i, (img_path, _) in enumerate(current_batch):
        current_batch[i] = (img_path, (x_offset,0))
        x_offset += THUMB_SIZE

    cv2.imshow("Batch Viewer", canvas)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        index += BATCH_SIZE
        continue

cv2.destroyAllWindows()
print("Batch inspection completed.")
