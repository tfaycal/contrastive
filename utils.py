from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDatasetPair(Dataset):
    """Custom Dataset for contrastive learning."""

    def __init__(self, data, targets, transform=None, target_transform=None):
        """
        Args:
            data (list or numpy array): List or array of images.
            targets (list or numpy array): List or array of labels.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on a label.
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        # Define the classes based on unique targets
        self.classes = np.unique(self.targets).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        
        img=Image.open(img_path)
        
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# Assuming your images are stored in /content/data/train/BRCA and /content/data/train/Other
train_positive_dir = 'data/train/BRCA'
train_negative_dir = 'data/train/Other'
test_positive_dir = 'data/test/BRCA'
test_negative_dir = 'data/test/Other'

# Assuming you have lists of file paths for the images
train_positive_images = [os.path.join(train_positive_dir, file) for file in os.listdir(train_positive_dir)]
train_negative_images = [os.path.join(train_negative_dir, file) for file in os.listdir(train_negative_dir)]
test_positive_images = [os.path.join(test_positive_dir, file) for file in os.listdir(test_positive_dir)]
test_negative_images = [os.path.join(test_negative_dir, file) for file in os.listdir(test_negative_dir)]

# Combine positive and negative lists with labels (1 for BRCA, 0 for Other)
train_images = train_positive_images + train_negative_images
train_labels = [1] * len(train_positive_images) + [0] * len(train_negative_images)
test_images = test_positive_images + test_negative_images
test_labels = [1] * len(test_positive_images) + [0] * len(test_negative_images)

# Split positive samples into train and test
_,positive_mem = train_test_split(test_positive_images, test_size=0.01, random_state=42)

# Split negative samples into train and test
_ ,negative_meme= train_test_split(test_negative_images, test_size=0.01, random_state=42)
mem_images=positive_mem+negative_meme
mem_labels = [1] * len(positive_mem) + [0] * len(negative_meme)
