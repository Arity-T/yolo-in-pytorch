import cv2
from torch.utils.data import Dataset as BaseDataset


def read_annot_file(path):
    """Reads annotations from file in YOLO format.

    Each line in file should look like:
    <object-class> <x> <y> <width> <height>

    Args:
        path (str): Path to the file

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
                annotations (see convert_voc_labels.py)
            augmentations (callable, optional): Albumentations augmentation pipeline or
                custom function/transform with same interface
            transforms (callable, optional): A function/transform that takes in an numpy
                array and returns a transformed version
            grid_size (int, optional): YOLO hyperparameter (see paper for details).
                Defaults to 7
            number_of_classes (int, optional): Number of classes in dataset.
                Defaults to 20
            read_annots_once (bool, optional): If True loads annotations in RAM during
                dataset initialization. Otherwise reads annotation file  every time
                __getitem__ is called. Defaults to True
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
            with open(path_to_set) as image_set_file:
                for line in image_set_file.read().strip().split("\n"):
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
        """Returns number of images in dataset"""
        return len(self.img_pathes)

    def __getitem__(self, index):
        """Returns image after transforms and corresponding grid"""
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
