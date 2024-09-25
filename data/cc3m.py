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

raw_dataset = load_dataset('./cc12m')['train']

Image.MAX_IMAGE_PIXELS = None
