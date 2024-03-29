
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from torch import Tensor, from_numpy, logical_not
from torch import stack as torch_stack
from torch import zeros as torch_zeros
from torchvision.datasets import VisionDataset

from lemanchot.rle import decode_rle
from lemanchot.dataset.unlabeled import ImageDataset

class JSONDataset(VisionDataset):
    """
    Dataset Class to handle RLEs encoded masks in JSON files.
    """

    def __init__(
        self,
        root: str,
        folder_name: str,
        classes: Set,
        transforms: Optional[Callable] = None,
    ):
        """
        Dataset Class to handle RLEs encoded masks in JSON files.
        Args:
            root (str): Root folder path of the dataset.
            folder_name (str): Directory name of the folder containint the JSON files.
            transforms (Optional[Callable], optional):
                Transformations to be applied once the targets are decoded. Defaults to None.
        Raises:
            ValueError: If the paths are not found.
        """
        super().__init__(root, transforms)
        self.folder_name = folder_name
        self.classes = classes

        if not os.path.isdir(self.root):
            raise ValueError("The dataset directory does not exist or is not valid!")

        self.paths = [str(p) for p in list(Path(root).rglob(f"{folder_name}/*.json"))]

    def __len__(self) -> int:
        return len(self.paths)

    def JSON2ClassMap(self, input: Dict) -> Dict:
        """
        Decoded and encoded RLE generated by the `generateJSON` function.

        Notes:
            Class indexes in the returned Array are in order of appearance in the JSON file.
            Annotations of the same class are added to the same index.

        Args:
            input (Dict): Decoded JSON file in a dictionary format.
            filter (Callable): Function used to filter and order decoded classes.
                Must take class set as input and return filtered class set as output.

        Returns:
            Dict: Decoded tensor in one-hot format class: tensor(W, H)
        """
        height = input["height"]
        width = input["width"]
        layers = {}
        for cl, ann in input["annotations"].items():
            if ann.get("data", False):
                layers[cl.lower()] = from_numpy(
                    decode_rle(ann["data"]).reshape(height, width, 4)[:, :, 3]
                )

        return layers

    def __getitem__(self, index: int) -> Tuple[str, Tensor]:
        """
        Return a path
        Args:
            index (int): Index of the file to be decoded.
        Returns:
            Tuple[str, Tensor]: Path of the decoded JSON and the decoded mask.
        """
        path = self.paths[index]
        with open(path, "r") as f:
            data = json.load(f)

        size = (data["height"], data["width"])
        target = self.JSON2ClassMap(data)
        target = torch_stack(
            [target.get(c, torch_zeros(size)) for c in self.classes.keys()], dim=0
        )

        if "background"in self.classes.keys():
            target[self.classes.background, ...] = logical_not(target.sum(dim=0))

        if self.transforms is not None:
            target = self.transforms(target)

        return path, target


class SegmentationDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        classes: List,
        img_folder: str = "img",
        img_ext: str = ".jpg",
        gt_folder: str = "gt",
        input_transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        both_transforms: Optional[Callable] = None,
    ):
        """
        Dataset Class to handle image files.
        Args:
            root (str): Root folder path of the dataset.
                Samples are placed in a folder in root/imgs and targets are in root/gts.
                Targets take priority when loading a sample, meaning the path of the
                samples is generated by replacing `gts` with `imgs`. Concequently, targets
                and samples share names and image format.
            gt_folder (str): Name of the ground truth folder.
            *_transform (Callable): Collection of transformations to be applied to the inputs
                targets, separetly of together.
        Raises:
            ValueError: If the paths are not found.
        """
        super().__init__(root)

        self.gt_dataset = JSONDataset(root, gt_folder, classes, target_transforms)
        img_paths = [
            p.replace(gt_folder, img_folder).replace(".json", img_ext)
            for p in self.gt_dataset.paths
        ]
        self.samples_dataset = ImageDataset(
            paths=img_paths, transforms=input_transforms
        )
        self.both_transforms = both_transforms

    def __len__(self) -> int:
        return len(self.gt_dataset)

    def __getitem__(self, index: int):

        path, sample = self.samples_dataset[index]
        _, target = self.gt_dataset[index]

        if self.both_transforms is not None:
            sample, target = self.both_transforms(sample, target)

        return sample, target