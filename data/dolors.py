import json
import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from data import load_dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class CC12M(Dataset):
    def __init__(self, path_or_name='nebula/cc12m', split='train', transform=None, single_caption=True):
        self.raw_dataset = load_dataset(path_or_name)[split]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to 224x224 pixels
                transforms.CenterCrop(224),  # Center crop to 224x224 pixels
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(  # Normalize with mean and std
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transforms.Compose(transform)

        self.single_caption = single_caption
        self.length = len(self.raw_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = self.raw_dataset[index]
        caption = item['txt']
        with io.BytesIO(item['webp']) as buffer:
            image = Image.open(buffer).convert('RGB')
            if self.transform:
                image = self.transform(image)
        del item
        return image, caption

import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np

from PIL import Image


Split = Literal["train", "valid", "test"]


class PathDataset:
    path_images: Path
    split: Split

    def __post_init__(self):
        self.files = self.load_filelist(self.split)

    def load_filelist(self, split):
        return os.listdir(self.path_images / split)

    def get_file_name(self, i):
        return Path(self.files[i]).stem

    def get_image_path(self, i):
        return str(self.path_images / self.split / self.files[i])

    def __len__(self):
        return len(self.files)


class WithMasksPathDataset(PathDataset):
    def __init__(self, path_images, path_masks, split):
        self.path_masks = path_masks
        super().__init__(path_images, split)

    def get_mask_path(self, i):
        return str(self.path_masks / self.split / self.files[i])

    @staticmethod
    def load_mask_keep(path):
        """Assumes that the in the original mask:

        - `255` means unchanged content, and
        - `0` means modified content.

        """
        mask = np.array(Image.open(path))
        mask = 1 - (mask[:, :, 0] == 255).astype("float")
        return mask

    def load_mask(self, i):
        return self.load_mask_keep(self.get_mask_path(i))

    def __len__(self):
        return len(self.files)


# § · Real datasets


class CelebAHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/celebahq/real")
        super().__init__(path_images=path_images, split=split)


class FFHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/ffhq/real")
        super().__init__(path_images=path_images, split=split)


# § · Fake datasets: Fully-manipulated
class P2CelebAHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/celebahq/fake/p2")
        super().__init__(path_images=path_images, split=split)


class P2FFHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/ffhq/fake/p2")
        super().__init__(path_images=path_images, split=split)


# § · Fake datasets: Partially-manipulated
class RepaintP2CelebAHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/repaint-p2")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class RepaintP2FFHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/ffhq/fake/repaint-p2")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )

class RepaintP2CelebAHQ9KDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/repaint-p2-9k")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class RepaintLDMCelebAHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/ldm")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class LamaDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/lama")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class PluralisticDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/pluralistic")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class ConcatDataset:
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

    def _get_dataset_index_and_offset(self, i):
        for j, length in enumerate(self.lengths):
            if i < length:
                return j, i
            i -= length
        raise IndexError

    def get_image_path(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j].get_image_path(i)

    def get_mask_path(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j].get_mask_path(i)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j][i]


from dolos.data import (
    CelebAHQDataset,
    P2CelebAHQDataset,
    RepaintP2CelebAHQ9KDataset,
)




def get_datapipe(config, split):
    load_image = config["load-image"]
    batch_size = config.get("batch-size", BATCH_SIZE)

    def get_sample(dataset, label, i):
        return {
            "image": load_image(dataset, i, split),
            "label": torch.tensor(label),
        }

    dataset_real = config["dataset-real"](split)
    dataset_fake = config["dataset-fake"](split)

    datapipe_real = SequenceWrapper(range(len(dataset_real)))
    datapipe_real = datapipe_real.map(partial(get_sample, dataset_real, 0))

    datapipe_fake = SequenceWrapper(range(len(dataset_fake)))
    datapipe_fake = datapipe_fake.map(partial(get_sample, dataset_fake, 1))

    datapipe = datapipe_real.concat(datapipe_fake)

    if split == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()

    # datapipe = datapipe.header(64)
    datapipe = datapipe.batch(batch_size)
    datapipe = datapipe.collate()

    return datapipe


def get_datapipe(config, split, transform_mask):
    dataset = config["dataset-class"](split)
    load_image = config["load-image"]

    def get_sample(i):
        image = load_image(dataset, i)
        mask = transform_mask(imread(dataset.get_mask_path(i)))
        mask = (mask < 0.5).float()
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(1),
        }

    datapipe = SequenceWrapper(range(len(dataset)))
    datapipe = datapipe.map(get_sample)

    if split == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()

    datapipe = datapipe.batch(BATCH_SIZE)
    datapipe = datapipe.collate()

    return datapipe

