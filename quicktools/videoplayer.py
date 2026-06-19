import cv2
import torch
import time
import numpy as np
import yaml

from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_augment import letterbox

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"           # Save output video
WEIGHTS_PATH = "weights/best_ckpt.pt"
DATA_YAML = "data/data.yaml"
IMG_SIZE = 640                        # YOLOv6 input size
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SKIP = 2                        # Detect every 2 frames
SCALE_FACTOR = 0.5                     # Scale video frames for faster processing

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
with open(DATA_YAML, "r") as f:
    data_dict = yaml.safe_load(f)
names = data_dict["names"]
print(f"[INFO] Classes: {names}")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("[INFO] Loading YOLOv6 model...")
model = DetectBackend(WEIGHTS_PATH, device=DEVICE)
model.eval()
stride = model.stride
print("[INFO] Model loaded successfully")

# -----------------------------
# VIDEO CAPTURE & WRITER
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Could not open video"

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE_FACTOR)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE_FACTOR)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (frame_width, frame_height))

frame_id = 0
cached_pred = None
cached_ratio_pad = None

print("[INFO] Starting video inference...")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached")
        break

    frame_id += 1

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    img0 = frame_resized.copy()

    # -----------------------------
    # DETECT ON FIRST FRAME OR EVERY FRAME_SKIP
    # -----------------------------
    if frame_id == 1 or frame_id % FRAME_SKIP == 0:
        # Letterbox for YOLOv6
        img, ratio, pad = letterbox(img0, IMG_SIZE, stride=stride, auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(DEVICE)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

        cached_pred = pred
        cached_ratio_pad = (ratio, pad)
    else:
        pred = cached_pred
        ratio, pad = cached_ratio_pad

    # -----------------------------
    # DRAW BOXES
    # -----------------------------
    if pred and pred[0] is not None:
        for *xyxy, conf, cls in pred[0]:
            # Undo letterbox scaling
            x1 = int((xyxy[0] - pad[0]) / ratio)
            y1 = int((xyxy[1] - pad[1]) / ratio)
            x2 = int((xyxy[2] - pad[0]) / ratio)
            y2 = int((xyxy[3] - pad[1]) / ratio)

            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    # -----------------------------
    # FPS DISPLAY
    # -----------------------------
    elapsed = time.time() - start_time
    fps = 1 / elapsed
    cv2.putText(frame_resized, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # -----------------------------
    # SHOW & SAVE
    # -----------------------------
    cv2.imshow("YOLOv6 Video Detection", frame_resized)
    out.write(frame_resized)

    # Real-time playback control
    wait_time = max(1, int((1/video_fps - elapsed)*1000))
    if cv2.waitKey(wait_time) & 0xFF == 27:  # ESC to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Detection completed. Output saved to {OUTPUT_PATH}")
