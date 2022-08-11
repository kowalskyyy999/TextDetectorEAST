import os
import glob
import numpy as np

from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence

from scripts.utils import *

class MSRADataset(Dataset):
    def __init__(self,
                 path,
                 test_size=None,
                 img_resize=(1600, 1200),
                 transform=None,
                 scale=0.25):

        self.path = path

        self.df = extract2df(path)
        self.df.drop_duplicates(['image', 'index', 'difficult'], inplace=True)

        self.image_ids = self.df.image.unique()
        self.test_size = test_size
        self.img_resize = img_resize
        self.transform = transform
        self.scale = scale
        self.mask_transform = T.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        ids = self.image_ids[idx]
        image = cv2.imread(os.path.join(self.path, f'{ids}.JPG')) # Load the image

        image_resize = cv2.resize(image, self.img_resize) # Resize image

        data = self.df[self.df['image'] == ids] # filtering data by image ids

        vertices, angles, mask = self.preprocessing(image, image_resize, data) # preprocessing to get vertices

        score_map, geo_map = get_score_geo(image_resize, vertices, angles, self.scale, self.img_resize) # get ground truth

        score_tensor = torch.Tensor(score_map).permute(2, 0, 1)
        geo_tensor = torch.Tensor(geo_map).permute(2, 0, 1)

        # mask shape (H, W)
        mask_tensor = self.mask_transform(mask)
        # mask_tensor shape (1, H, W)

        if self.transform:
            image = self.transform(image_resize)

        return image, mask_tensor, score_tensor, geo_tensor

    def preprocessing(self, img, img_resize, data):
        h, w, _ = img.shape
        h_r, w_r, _ = img_resize.shape
        mask = np.zeros((h_r, w_r))

        vertices = []
        angles = []

        for d in data.values:
            x, y, w, h = Resize(d, w, h, w_r, h_r)
            cx = int(w) // 2 + int(x)
            cy = int(h) // 2 + int(y)

            corner = get_corners([x, y, w, h])
            corner = move_points(corner, cx, cy, d[8])

            points = np.array(corner).reshape((-1, 1, 2))

            mask = get_mask(mask, points)

            vertices.append(get_vertices(corner))
            angles.append(d[8])

        return vertices, angles, mask
    
def collate_fn(batch):
    (image, mask, score, geo) = list(zip(*batch))
    image_pad = pad_sequence(image, batch_first=True)
    score_pad = pad_sequence(score, batch_first=True)
    geo_pad = pad_sequence(geo, batch_first=True)
    
    return image_pad, score_pad, geo_pad

if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor()
    ])
    train_path = "/home/kowlsss/Desktop/tutorial/torch/east/data/MSRA-TD500/train"
    train_dataset = MSRADataset(path=train_path, transform=transform)
    
    for data in train_dataset:
        print("Geometric GT shape:", data[-1].shape)
        print("Score GT shape:", data[2].shape)
        # print(data[0].shape)
        # print(data[-1].shape)
        break
