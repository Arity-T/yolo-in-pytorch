import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import default_collate
from tqdm import tqdm

import utils


def read_annot_file(path):
    """Reads annotations from file in YOLO format.

    Each line in file should look like:
    <object-class> <x> <y> <width> <height>

    Args:
        path (str): Path to the file.

    Returns:
        tuple: (list of class labels, list of bboxes)
    """
    labels = []
    bboxes = []

    with open(path) as annot_file:
        for line in annot_file.read().strip().split("\n"):
            line = line.split()
            labels.append(int(line[0]))
            bboxes.append([float(bbox_param) for bbox_param in line[1:5]])

    return labels, bboxes


def collate_fn(batch):
    """Pass this function as collate_fn argument of the DataLoader"""
    images, annotations = zip(*batch)
    return default_collate(images), annotations


class Dataset(BaseDataset):
    def __init__(
        self,
        img_sets,
        augmentations=None,
        transforms=None,
        grid_size=7,
        number_of_classes=20,
        read_annots_once=True,
    ):
        """Reads dataset in YOLO format.

        Args:
            img_set (list of strings): List of pathes to files with pathes to images and
                annotations (see convert_voc_labels.py).
            augmentations (callable, optional): Albumentations augmentation pipeline or
                custom function/transform with same interface. When using albumentations
                Compose, pass albumentations.BboxParams("yolo", label_fields=["labels"]))
                as bbox_params argument of Compose.
            transforms (callable, optional): A function/transform that takes in an numpy
                array and returns a transformed version.
            grid_size (int, optional): YOLO hyperparameter (see paper for details).
                Defaults to 7.
            number_of_classes (int, optional): Number of classes in dataset.
                Defaults to 20.
            read_annots_once (bool, optional): If set to True loads annotations in RAM
                during dataset initialization. Otherwise reads annotation file every
                time __getitem__ is called. Defaults to True.
        """
        self.augmentations = augmentations
        self.transforms = transforms
        self.grid_size = grid_size
        self.number_of_classes = number_of_classes

        # Reading files with image sets
        self.img_pathes = []
        self.annot_pathes = []
        self.annotations = []

        for path_to_set in img_sets:
            if read_annots_once:
                print(f"Reading annnotations for {path_to_set} image set")

            with open(path_to_set) as image_set_file:
                for line in tqdm(
                    image_set_file.read().strip().split("\n"),
                    disable=(not read_annots_once),
                ):
                    splitted_line = line.strip().split(" ")
                    assert len(splitted_line) == 2, (
                        f"Something went wrong during reading image set file '{path_to_set}'. "
                        + f"Extra spaces in line: '{line}'"
                    )
                    self.img_pathes.append(splitted_line[0])

                    if read_annots_once:
                        self.annotations.append(read_annot_file(splitted_line[1]))
                    else:
                        self.annot_pathes.append(splitted_line[1])

    def __len__(self):
        """Returns number of images in dataset."""
        return len(self.img_pathes)

    def __getitem__(self, index):
        """Returns image after transforms and corresponding annotations."""
        # Read image
        img = cv2.cvtColor(cv2.imread(self.img_pathes[index]), cv2.COLOR_BGR2RGB)

        # Read annotations
        if self.annotations:
            labels, bboxes = self.annotations[index]
        else:
            labels, bboxes = read_annot_file(self.annot_pathes[index])

        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=img, bboxes=bboxes, labels=labels)
            img = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, {"bboxes": bboxes, "labels": labels}


def add_activations(model, activation, *args, **kwargs):
    """Adds activation functions after all torch.nn.Conv2d layers.

    Args:
        model (torch.nn.Sequential or list): A convolution model or a list of its layers
            to which activation functions should be added.
        activation: An activation function that will be initialized with the given
            parameters and added to the model layers, e.g. torch.nn.LeakyReLU.
        *args: Positional arguments to initialize the activation function.
        **kwargs: Keyword arguments to initialize the activation function.
    Returns:
        torch.nn.Sequential: New model.
    """
    new_model = []

    for l in model:
        new_model.append(l)

        if isinstance(l, nn.Conv2d):
            new_model.append(activation(*args, **kwargs))

    return nn.Sequential(*new_model)


class Backbone(nn.Module):
    def __init__(self):
        """Original YOLOv1 backbone."""
        super().__init__()

        self.layers = [
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1, 1, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
        ]

        self.layers = add_activations(self.layers, nn.LeakyReLU, negative_slope=0.1)

    def forward(self, batch):
        return self.layers(batch)


class Model(nn.Module):
    def __init__(
        self, backbone=None, grid_size=7, number_of_bboxes=2, number_of_classes=20
    ):
        """Creates YOLOv1 model.

        Args:
            backbone (callable, optional): Backbone model. If not specified default
                backbone will be used (see Figure 3 in paper for details).
            grid_size (int, optional): YOLO hyperparameter (see paper for details).
                Defaults to 7.
            number_of_bboxes (int, optional): Number of bounding boxes to predict per
                grid cell. Defaults to 2.
            number_of_classes (int, optional): Number of classes. Defaults to 20.
        """
        super().__init__()
        self.grid_size = grid_size
        self.number_of_bboxes = number_of_bboxes
        self.preds_per_cell = number_of_bboxes * 5 + number_of_classes

        self.backbone = backbone if backbone else Backbone()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * self.preds_per_cell),
        )

    def forward(self, batch):
        batch = self.backbone(batch)
        batch = self.fc_layers(batch)
        batch = batch.reshape(-1, self.grid_size, self.grid_size, self.preds_per_cell)
        return batch

    @torch.no_grad()
    def predict(self, batch, threshold=0.5):
        self.eval()
        predicted_grids = self(batch)

        cell_size = 1 / self.grid_size
        prediction = []

        for grid in predicted_grids:
            current_pred = {
                "bboxes": [],
                "labels": [],
                "class_scores": [],
                "bbox_scores": [],
            }

            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    cell = grid[row, col]

                    # Find prediction with the highest confidence
                    max_conf_bbox = None
                    for bbox_i in range(self.number_of_bboxes):
                        if (
                            max_conf_bbox is None
                            or cell[(bbox_i + 1) * 5 - 1] > max_conf_bbox[-1]
                        ):
                            max_conf_bbox = cell[bbox_i * 5 : (bbox_i + 1) * 5]

                    # Skip empty cells
                    if max_conf_bbox[-1] < threshold:
                        continue

                    # Normalize bounding box parameters
                    x, y, w, h, conf = max_conf_bbox

                    x = (x + col) * cell_size
                    x = float(min(1, max(0, x)))

                    y = (y + row) * cell_size
                    y = float(min(1, max(0, y)))

                    w = float(min(1, max(0, w**2)))

                    h = float(min(1, max(0, h**2)))

                    conf = float(min(1, max(0, conf)))

                    # Find class with the highest confidence score
                    class_conf, class_index = cell[-20:].max(-1)
                    class_conf = float(min(1, max(0, class_conf)))
                    class_index = int(class_index)

                    # Save prediction for current cell
                    current_pred["bboxes"].append([x, y, w, h])
                    current_pred["labels"].append(class_index)
                    current_pred["class_scores"].append(class_conf)
                    current_pred["bbox_scores"].append(conf)

            prediction.append(current_pred)

        return prediction


class Loss(nn.Module):
    def __init__(self, labmda_coord=5.0, labmda_noobj=0.5):
        """Implementation of the original YOLOv1 loss function.

        Args:
            labmda_coord (float, optional): Coefficient for bounding box coordinate
                predictions (see paper for details). Defaults to 5.
            labmda_noobj (float, optional): Coefficient for confidence predictions for
                bounding boxes that don't contain objects (see paper for details).
                Defaults to 0.5.
        """
        super().__init__()

        self.lambda_coord = labmda_coord
        self.lambda_noobj = labmda_noobj

    def forward(self, pred, gt):
        """Calculates loss.

        Args:
            pred (torch.tensor): Tensor predicted by the model.
            gt (list of dicts): List of ground truth bboxes with labels.

        Returns:
            torch.tensor: Tensor of size 1.
        """
        loss = 0
        cell_size = 1 / pred.shape[1]

        for ex_i in range(len(gt)):
            cells_with_obj = []

            for (x, y, w, h), label in zip(gt[ex_i]["bboxes"], gt[ex_i]["labels"]):
                # Find grid cell responsible for predicting current bbox
                row, col = int(x // cell_size), int(y // cell_size)

                # Yolo can predict only one object per grid cell so some bboxes will not
                # be taken into account
                if (row, col) in cells_with_obj:
                    continue

                cells_with_obj.append((row, col))

                # Predict coordinates relative to the bounds of the grid cell
                x = x / cell_size - col
                y = y / cell_size - row

                # Predict the square root of the bounding box width and height
                w **= 0.5
                h **= 0.5

                # Choose predicted bbox with the highest IOU with the ground truth
                current_pred = pred[ex_i, row, col]
                bbox1_iou = utils.iou(current_pred[:4], (x, y, w, h))
                bbox2_iou = utils.iou(current_pred[5:9], (x, y, w, h))

                pred_bbox, pred_bbox_iou = (
                    (current_pred[:5], bbox1_iou)
                    if bbox1_iou > bbox2_iou
                    else (current_pred[5:10], bbox2_iou)
                )

                # Coordinates and size loss
                loss += self.lambda_coord * F.mse_loss(
                    pred_bbox[:4],
                    torch.tensor([x, y, w, h], dtype=torch.float, device=pred.device),
                    reduction="sum",
                )

                # Confidence loss
                loss += F.mse_loss(
                    pred_bbox[4],
                    torch.tensor(pred_bbox_iou, dtype=torch.float, device=pred.device),
                )

                # Classification loss
                loss += F.mse_loss(
                    current_pred[-20:],
                    F.one_hot(torch.tensor(label), 20).to(
                        device=pred.device, dtype=torch.float
                    ),
                    reduction="sum",
                )

            # No object loss
            confidence_preds = pred[ex_i][..., [4, 9]]
            # Don't count no object loss for cells with objects
            confidence_preds[tuple(zip(*cells_with_obj))] *= 0
            loss += self.lambda_noobj * F.mse_loss(
                confidence_preds, torch.zeros_like(confidence_preds), reduction="sum"
            )

        return loss / len(gt)
