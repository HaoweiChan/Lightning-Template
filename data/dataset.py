from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Initialize your dataset here
        
    def __len__(self):
        # Return the size of dataset
        pass
        
    def __getitem__(self, idx):
        # Return one item of data
        pass 