from hashlib import md5

import cv2
import numpy as np
import torch


@torch.no_grad()
def iou(bbox1, bbox2):
    """Calculates Intersection Over Union for two bounding boxes in YOLO format."""
    x1, y1, w1, h1 = bbox1.tolist() if isinstance(bbox1, torch.Tensor) else bbox1
    x2, y2, w2, h2 = bbox2.tolist() if isinstance(bbox2, torch.Tensor) else bbox2

    if w1 < 0 or h1 < 0 or w2 < 0 or h2 < 0:
        return 0

    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection

    if union == 0:
        return 0.0

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


def draw_bboxes(img, bboxes, class_names=None):
    """Draws bounding boxes on copy of the input image."""
    img = img_to_numpy(img)

    if not class_names:
        class_names = open("voc_classes.txt").read().strip().split("\n")

    for bbox in bboxes:
        bbox_color = to_color(bbox[4]) if len(bbox) > 4 else to_color(bbox)

        x_min, y_min, x_max, y_max = bbox_to_corners(
            bbox[:4], img.shape[1], img.shape[0]
        )
        cv2.rectangle(
            img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=bbox_color, thickness=2
        )

        if len(bbox) > 4:
            text = (
                class_names[bbox[4]]
                + " "
                + " ".join([str(round(score, 2)) for score in bbox[5:]])
            )
            img = put_text(img, text, x_min, y_min, bbox_color)

    return img


def nms(predictions, iou_threshold=0.5):
    """Non Max Suppression algorithm implementation.

    Args:
        predictions (list): A list of predictions where each prediction is a list of
            bounding boxes. Each bounding box is represented as a tuple that looks like
            (x, y, w, h, class, class-specific confidence score, ...)
        iou_threshold (float, optional): The iou threshold for suppressing extra
            bounding boxes. A lower threshold means stricter filtering. Defaults to 0.5.

    Returns:
        A list of filtered predictions in the same format as the input.
    """
    result = []

    for bboxes in predictions:
        bboxes = sorted(bboxes, key=lambda x: x[5])
        result.append([])

        while bboxes:
            current_bbox = bboxes.pop()

            bboxes = [
                bbox
                for bbox in bboxes
                if current_bbox[4] != bbox[4]
                or iou(current_bbox[:4], bbox[:4]) < iou_threshold
            ]

            result[-1].append(current_bbox)

    return result


def compute_map(predictions, annots, iou_threshold=0.5, number_of_classes=20):
    """Computes mAP metric used in PASCAL VOC dataset.

    The official documentation explaining PASCAL VOC criteria for object detection
    metrics can be accessed by this link
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00054000000000000000

    Python implementation https://github.com/Cartucho/mAP

    Args:
        predictions (list): A list of predictions where each prediction is a list of
            bounding boxes. Each bbox is represented as a tuple that looks like
            (x, y, w, h, class, score).
        annots (list): A list ground truth where each element is a list of bboxes. Each
            bbox is represented as a tuple that looks like (x, y, w, h, class).
        iou_threshold (float, optional): Consider a prediction as True Positive only
            when IOU with ground truth greater than this threshold. Defaults to 0.5.
        number_of_classes (int, optional): Number of classes in dataset. Defaults to 20.

    Returns:
        float: Calculated mAP value.
    """
    bboxes_by_class = [[] for _ in range(number_of_classes)]
    num_of_bboxes_by_class = [0] * number_of_classes

    # For cases when not all classes are presented in test data
    is_class_presented = [0] * number_of_classes

    # Iterate over images and save true positives
    for pred_bboxes, gt_bboxes in zip(predictions, annots):
        # Sort predicted bboxes by confidence score in descending order
        pred_bboxes = sorted(pred_bboxes, key=lambda x: x[5], reverse=True)

        for gt_bbox in gt_bboxes:
            is_class_presented[gt_bbox[4]] = 1
            num_of_bboxes_by_class[gt_bbox[4]] += 1

            for pred_bbox in pred_bboxes:
                # Compare class and IOU of predicted and ground truth bboxes
                if (
                    gt_bbox[4] == pred_bbox[4]
                    and iou(gt_bbox[:4], pred_bbox[:4]) >= iou_threshold
                ):
                    bboxes_by_class[pred_bbox[4]].append([pred_bbox[5], True])
                    pred_bboxes.remove(pred_bbox)
                    break

        # Consider the remaining predictions as false positives
        for pred_bbox in pred_bboxes:
            bboxes_by_class[pred_bbox[4]].append([pred_bbox[5], False])

    # Calculate average precision (AP) values for all classes
    ap_by_class = []

    # Iterate over classes
    for pred_list, num_of_gt in zip(bboxes_by_class, num_of_bboxes_by_class):
        if num_of_gt == 0:
            ap_by_class.append(0)
            continue

        average_precision = 0

        last_precision = 0
        last_recall = 0

        recall = 0

        true_positives_count = 0

        for i, (_, is_true_positive) in enumerate(
            # Sort predicted bboxes by confidence score in descending order
            sorted(pred_list, key=lambda x: x[0], reverse=True)
        ):
            true_positives_count += is_true_positive
            precision = true_positives_count / (i + 1)
            recall = true_positives_count / num_of_gt

            if precision < last_precision:
                average_precision += (recall - last_recall) * last_precision
                last_recall = recall

            last_precision = precision

        average_precision += (recall - last_recall) * last_precision

        ap_by_class.append(average_precision)

    return sum(ap_by_class) / sum(is_class_presented), ap_by_class
