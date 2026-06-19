import os

img_dir = r"D:/detector/NotMonkey/images/train"
lbl_dir = r"D:/detector/NotMonkey/labels/train"

imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
lbls = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir)}

missing = imgs.keys() - lbls.keys()

print("Images without matching labels:", len(missing))
print("Example mismatches:", list(missing)[:10])


