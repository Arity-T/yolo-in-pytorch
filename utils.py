from hashlib import md5

import cv2
import numpy as np
import torch


@torch.no_grad()
def iou(bbox1, bbox2):
    """Calculates Intersection Over Union for two bounding boxes in YOLO format."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    if w1 < 0 or h1 < 0 or w2 < 0 or h2 < 0:
        return 0

    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection

    if union == 0:
        return 0

    return intersection / union


def bbox_to_corners(bbox, img_width, img_height):
    """Converts bbox from YOLO to VOC format."""
    x, y, w, h = bbox

    x_min = int((x - w / 2) * img_width)
    y_min = int((y - h / 2) * img_height)
    x_max = int((x + w / 2) * img_width)
    y_max = int((y + h / 2) * img_height)

    return x_min, y_min, x_max, y_max


def img_to_numpy(img):
    """Converts copy of img to opencv and matplotlib compitable format."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().clone().numpy()
    else:
        img = img.copy()

    if img.max() <= 1:
        img *= 255

    img = img.astype(np.uint8).squeeze()

    if img.shape[0] == 3:
        # OpenCV needs a contiguous array
        img = np.ascontiguousarray(img.transpose(1, 2, 0))

    return img


def to_color(obj):
    """Converts any serializable object to color tuple."""
    h = md5(str(obj).encode()).hexdigest()
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def put_text(img, text, x, y, bg_color, left_top_origin=True):
    """Draws text with backgroung on copy of the input image and returns it."""
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


def draw_bboxes(img, bboxes, labels=None, scores=None):
    """Draws bounding boxes, labels and confidence scores on copy of the input image."""
    img = img_to_numpy(img)

    for i, bbox in enumerate(bboxes):
        bbox_color = to_color(labels[i]) if labels else to_color(bbox)

        x_min, y_min, x_max, y_max = bbox_to_corners(bbox, img.shape[1], img.shape[0])
        cv2.rectangle(
            img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=bbox_color, thickness=2
        )

        if labels:
            text = f"{labels[i]} {scores[i]:.2f}" if scores else labels[i]
            img = put_text(img, text, x_min, y_min, bbox_color)

    return img
