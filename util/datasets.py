import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, csv_file, model_set='diffusion_set', set='train', x_transform=None, y_transform=None, feature_names=[], n="all"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file)
        df = df[df[model_set] == set]
        if n != "all":
          df = df.head(n)
        self.metadata_frame = df
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.feature_names = feature_names

    def __len__(self):
        return len(self.metadata_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path =  self.metadata_frame.iloc[idx].file_path
        image = Image.open(file_path)

        # Get attributes from the DataFrame
        attributes = self.metadata_frame.iloc[idx][self.feature_names].values.astype(np.float32)
        attributes = torch.from_numpy(attributes)
    
        if self.x_transform:
          image = self.x_transform(image)

        # if self.y_transform:
        #   attributes = self.y_transform(attributes)

        return image, attributes