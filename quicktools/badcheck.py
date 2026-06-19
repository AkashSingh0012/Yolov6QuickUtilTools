import os

LABEL_DIR = "D:/detector/YOLOv6/data/labels/train"
bad = 0
total = 0

for f in os.listdir(LABEL_DIR):
    if not f.endswith(".txt"):
        continue
    total += 1
    path = os.path.join(LABEL_DIR, f)
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
        if not lines:
            bad += 1
            continue
        for l in lines:
            parts = l.split()
            if len(parts) != 5:
                bad += 1
                break
            cls, x, y, w, h = map(float, parts)
            if not (0 <= cls < 1 and 0 < x < 1 and 0 < y < 1 and 0 < w < 1 and 0 < h < 1):
                bad += 1
                break

print(f"Total label files: {total}")
print(f"Bad/empty labels: {bad}")
