import os
LABEL_DIR = r"D:/detector/YOLOv6/data/labels/train"

bad_cls = set()

for f in os.listdir(LABEL_DIR):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(LABEL_DIR, f)) as fh:
        for line in fh:
            cls = float(line.split()[0])
            if cls != int(cls) or int(cls) != 0:
                bad_cls.add(cls)

print("Invalid class IDs found:", bad_cls)

