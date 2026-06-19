import os
import xml.etree.ElementTree as ET

# ---------------------------
# CONFIG
# ---------------------------
XML_DIR = r"D:/detector/MonkeyImg/labels/valxml" 
""
LABEL_DIR = r"dataset/labels/val"

# Class names in correct order
CLASSES = [
    "monkey"
]

os.makedirs(LABEL_DIR, exist_ok=True)


def convert_bbox(size, box):
    """
    Convert VOC bbox to YOLO format
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin, xmax, ymin, ymax = box

    x_center = ((xmin + xmax) / 2.0) * dw
    y_center = ((ymin + ymax) / 2.0) * dh
    width = (xmax - xmin) * dw
    height = (ymax - ymin) * dh

    return x_center, y_center, width, height


def convert_xml(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    with open(txt_path, "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text

            if cls not in CLASSES:
                print(f"[WARNING] Skipping unknown class: {cls}")
                continue

            cls_id = CLASSES.index(cls)

            xml_box = obj.find("bndbox")
            b = (
                float(xml_box.find("xmin").text),
                float(xml_box.find("xmax").text),
                float(xml_box.find("ymin").text),
                float(xml_box.find("ymax").text),
            )

            bb = convert_bbox((img_w, img_h), b)
            f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in bb)}\n")


# ---------------------------
# MAIN LOOP
# ---------------------------
for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(XML_DIR, xml_file)
    txt_file = os.path.splitext(xml_file)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_file)

    convert_xml(xml_path, txt_path)

print("✅ XML to YOLOv6 conversion completed.")
