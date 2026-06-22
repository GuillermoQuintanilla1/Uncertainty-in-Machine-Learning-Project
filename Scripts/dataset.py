from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class HAM10000Dataset(Dataset):

    def __init__(self,
                 dataframe,
                 transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = self.df.loc[idx, 'path']

        image = Image.open(img_path).convert("RGB")

        label = self.df.loc[idx,
                            'cell_type_idx']
        
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])

def create_dataloaders(
        train_df,
        val_df,
        test_df,
        batch_size=8):

    train_dataset = HAM10000Dataset(
        train_df,
        train_transform
    )

    val_dataset = HAM10000Dataset(
        val_df,
        test_transform
    )

    test_dataset = HAM10000Dataset(
        test_df,
        test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
