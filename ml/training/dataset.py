import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transform

class AnimalDataset(Dataset):
    def __init__(self, image_paths, class_mapping, transform=None):
        self.image_paths = image_paths
        self.class_mapping = class_mapping
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image, label = load_image_with_label(image_path, self.class_mapping)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_class_mapping():
    return {

    }

def get_idx_to_class():
    return {

    }

def get_image_paths(data_dir):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    paths = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                paths.append(full_path)
    return paths

def load_image_with_label(image_path, class_mapping):
    path_obj = Path(image_path)
    class_name = path_obj.parent-name.lower()

    label = class_mapping[class_name]

    image = Image.open(image_path).convert('RGB')

    return image, label