import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os


class MultiVarMeteorologicalDataset(Dataset):
    """
    A PyTorch Dataset for handling multi-variable meteorological data.
    The input is expected to be a 3D numpy array with shape [variables, longitude, latitude].
    
    Args:
        data_dir (str): Directory containing the numpy array files (.npy)
        transform (callable, optional): Optional transform to be applied to the data
        target_transform (callable, optional): Optional transform for the target
    """
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get list of sample files (assuming each sample is stored as a .npy file)
        # 接受的是处理好的ERA5数据集
        self.sample_files = [f for f in os.listdir(data_dir) 
                            if f.endswith('.npy') and not f.startswith('target_')]
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load input data (3D numpy array [variables, lon, lat])
        input_path = os.path.join(self.data_dir, self.sample_files[idx])
        input_data = np.load(input_path)
        
        # Load corresponding target
        base_name = os.path.splitext(self.sample_files[idx])[0]
        target_path = os.path.join(self.data_dir, f"target_{base_name}.npy")
        target = np.load(target_path)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target).float()
        
        # Apply transforms if specified
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)
            
        return input_tensor, target_tensor


def get_default_transform():
    """Returns a default transform for normalizing meteorological data"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0)  # Simple normalization
    ])