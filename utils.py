from hashlib import md5

import cv2
import numpy as np
import torch


def bbox_to_corners(bbox, img_width, img_height):
    """Converts bbox from YOLO to VOC format"""
    x, y, w, h = bbox

    x_min = int((x - w / 2) * img_width)
    y_min = int((y - h / 2) * img_height)
    x_max = int((x + w / 2) * img_width)
    y_max = int((y + h / 2) * img_height)

    return x_min, y_min, x_max, y_max


def img_to_numpy(img):
    """Converts img to opencv and matplotlib compitable format"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    if img.max() <= 1:
        img *= 255

    img = img.astype(np.uint8).squeeze()

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    return img.astype(np.uint8)


def to_color(obj):
    """Converts any serializable object to color tuple"""
    h = md5(str(obj).encode()).hexdigest()
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def put_text(img, text, x, y, bg_color, left_top_origin=True):
    """Draws text with backgroung on copy of the input image and returns it"""
    img = img.copy()

    text_params = {
        "text": str(text),
        "fontFace": cv2.FONT_HERSHEY_PLAIN,
        "fontScale": 1.2,
        "thickness": 1,
    }

    (text_w, text_h), text_bl = cv2.getTextSize(**text_params)

    if left_top_origin:
        y += text_h + text_bl // 2

    cv2.rectangle(
        img,
        pt1=(x, y - text_h - text_bl // 2),
        pt2=(x + text_w, y + text_bl // 2),
        color=bg_color,
        thickness=-1,
    )

    cv2.putText(img, org=(x, y), color=(255, 255, 255), **text_params)

    return img


def draw_bboxes(img, bboxes, labels=None):
    """Draws bounding boxes and labels on copy of the input image and returns it"""
    img = img_to_numpy(img).copy()

    for i, bbox in enumerate(bboxes):
        bbox_color = to_color(labels[i]) if labels else to_color(bbox)

        x_min, y_min, x_max, y_max = bbox_to_corners(bbox, img.shape[1], img.shape[0])
        cv2.rectangle(
            img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=bbox_color, thickness=2
        )

        if labels:
            img = put_text(img, labels[i], x_min, y_min, bbox_color)

    return img
