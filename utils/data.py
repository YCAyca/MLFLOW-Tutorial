import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import glob
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import make_grid 
import cv2


class Fruits_Dataset(Dataset):
    def __init__(self, image_paths, class_idx, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_idx = class_idx
         
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('/')[-2]
        label = self.class_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_dataset(data_path, generator=None):
    class_names = []
    images = []
    
    folders = os.listdir(data_path)
    for folder_name in folders:
        if folder_name == "inference_dataset":
            continue
        d = os.path.join(data_path, folder_name)
        class_names.append(folder_name)
        if os.path.isdir(d):
            for img_name in os.listdir(os.path.join(data_path,folder_name)):
                d = os.path.join(data_path, folder_name, img_name)
                images.append(d)
            
    class_idx = {j : i for i, j in enumerate(class_names)} #{'spor': 0, 'red_edition': 1}

    dataset_length = len(images)

    train_size = int(dataset_length * 6 / 10) # assign %60 of the images to the train dataset
    train_size = train_size - (train_size % 4) # be sure its divisible to batch size being at least 4, if its less (like 1), batch norm gives error
    test_size = int(dataset_length * 1 / 10) # assign %20 of the images to the test dataset
    val_size = dataset_length - train_size - test_size  # assign %30 of the images to the validation dataset

    
    print("Dataset includes", dataset_length, " images: ", train_size, " of them assigned into train", 
          test_size, " of them assigned into in test", val_size, " of them assigned into validation sub dataset")
    
    print("classnames", class_names)
    
    train_idx, test_idx, val_idx = random_split(images, [train_size, test_size, val_size], generator=generator) 

    train_list=[images[i] for i in train_idx.indices]
    test_list=[images[i] for i in test_idx.indices]
    val_list=[images[i] for i in val_idx.indices]
    
    return train_list, val_list, test_list, class_idx

