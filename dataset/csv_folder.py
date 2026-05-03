import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image

def find_classes(csv_file: Union[str, Path]):
    df = pd.read_csv(csv_file)
    classes = sorted(df['class'].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(csv_file: Union[str, Path]):
    df = pd.read_csv(csv_file)
    samples = []
    for _, row in df.iterrows():
        img_id = row['id']
        label = row['class']
        samples.append((img_id, label))
    return samples


class CSVFolder(Dataset):
    """
    A custom dataset class for a dataset where the images are arranged in this way by default: ::
    data_root/
        images/
            images/
                image_1.jpg
                image_2.jpg
                ...
        train.csv
        test.csv
            ...
    
    The train.csv should have the following format:
    id,class 

    """
    def __init__(
        self,
        data_root: Union[str, Path],
        transform: Optional[Callable] = None
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform

        csv_file = os.path.join(self.data_root, "train.csv")
        classes, class_to_idx = find_classes(csv_file)
        self.classes = classes
        self.class_to_idx = class_to_idx        
        
        samples = make_dataset(csv_file)
        self.samples = samples


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_id, label = self.samples[index]
        img_path = os.path.join(self.data_root, "images", "images", str(img_id) + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label_idx = self.class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
    


    





        
