import os

LABEL_DIR = r"D:/detector/YOLOv6/data/labels/train"

min_wh = 1e9
max_wh = 0

for f in os.listdir(LABEL_DIR):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(LABEL_DIR, f)) as fh:
        for line in fh:
            cls, x, y, w, h = line.split()
            cls = int(float(cls))
            x, y, w, h = map(float, (x, y, w, h))
            min_wh = min(min_wh, w, h)
            max_wh = max(max_wh, w, h)

print("Smallest w/h:", min_wh)
print("Largest w/h:", max_wh)
