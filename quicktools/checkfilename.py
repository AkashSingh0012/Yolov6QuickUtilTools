from pathlib import Path

img_dir = Path("Negative/images")
lbl_dir = Path("Negative/negative_labels")

imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}
lbls = {p.stem for p in lbl_dir.glob("*.txt")}

print("Images:", len(imgs))
print("Labels:", len(lbls))
print("Missing labels:", sorted(imgs - lbls)[:5])
