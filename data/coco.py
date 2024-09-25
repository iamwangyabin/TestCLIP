import json
import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class MSCOCO(Dataset):
    def __init__(self, path_or_name='clip-benchmark/wds_mscoco_captions2017', split='train', transform=None, single_caption=True):
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        self.raw_dataset = load_dataset(path_or_name, split=split)
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
        sample = self.raw_dataset[index]
        captions = sample['txt']
        captions = [caption.strip() for caption in captions.split('\n') if caption.strip()]

        if self.single_caption:
            caption = random.choice(captions)
        else:
            caption = captions

        image = sample['jpg'].convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption



