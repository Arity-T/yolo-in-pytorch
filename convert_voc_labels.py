"""Copied from https://pjreddie.com/media/files/voc_label.py with some changes

Converts annotations from VOC to YOLO format. By YOLO format I mean a txt file for each
image with a line for each ground truth object in the image that looks like:
<object-class> <x> <y> <w> <h>
Where <object-class> is the index of the class (see voc_classes.txt), <x> and <y> are 
coordinates of the center of bounding box, <w> and <h> are width and height of the
bounding box. <x>, <y>, <w>, <h> are normalized by the image width and height so that
they fall between 0 and 1.

Also creates a txt file for each image set (e.g. voc2007_train.txt) in which each line 
looks like:
<path-to-image> <path-to-annotations>

Run following commands to download VOC dataset and convert its annotations:
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
python convert_voc_labels.py
"""
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open(f"VOCdevkit/VOC{year}/Annotations/{image_id}.xml")
    out_file = open(f"VOCdevkit/VOC{year}/labels/{image_id}.txt", "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")


image_sets = [
    ("2012", "train"),
    ("2012", "val"),
    ("2007", "train"),
    ("2007", "val"),
    ("2007", "test"),
]

if __name__ == "__main__":
    with open("voc_classes.txt") as f:
        classes = f.read().strip().split()

    for year, image_set in tqdm(image_sets):
        os.makedirs(f"VOCdevkit/VOC{year}/labels/", exist_ok=True)

        with open(f"VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt") as f:
            image_ids = f.read().strip().split()

        with open(f"voc{year}_{image_set}.txt", "w") as list_file:
            for image_id in tqdm(image_ids, leave=False):
                list_file.write(
                    f"VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg VOCdevkit/VOC{year}/labels/{image_id}.txt\n"
                )
                convert_annotation(year, image_id)
