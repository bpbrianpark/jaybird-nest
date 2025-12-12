import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def get_train_transform():
    """Get training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=2, shuffle=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_class_mapping():
    return {
        'barred_owl': 0,
        'bear': 1,
        'coyote': 2,
        'hummingbird': 3,
        'osprey': 4,
        'sandhill_crane': 5
    }

def get_idx_to_class():
    return {
        0: 'barred_owl',
        1: 'bear',
        2: 'coyote',
        3: 'hummingbird',
        4: 'osprey',
        5: 'sandhill_crane'
    }

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
    class_name = path_obj.parent.name.lower()
    
    if class_name not in class_mapping:
        raise ValueError(f"Unknown class: {path_obj.parent.name}. Expected one of: {list(set(class_mapping.keys()))}")
    
    label = class_mapping[class_name]

    image = Image.open(image_path).convert('RGB')

    return image, label

def split_data(image_paths, class_mapping, train_ratio=0.7, val_ratio=0.15):
    for class_name, class_paths in image_paths.items():
        train, temp = train_test_split(class_paths, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test

if __name__ == "__main__":
    data_dir = "../photo_data"
    paths = get_image_paths(data_dir)
    mapping = get_class_mapping()
    idx_to_class = get_idx_to_class()
    
    print(f"Found {len(paths)} images")
    print(f"Classes: {list(idx_to_class.values())}")
    
    if paths:
        img, label = load_image_with_label(paths[0], mapping)
        class_name = idx_to_class[label]
        print(f"Loaded image: {paths[0]}")
        print(f"Class: {class_name}, Label: {label}")
    
    transform = get_train_transform()
    dataset = AnimalDataset(paths, mapping, transform=transform)
    print(f"\nDataset size: {len(dataset)}")
    img_tensor, label = dataset[0]
    print(f"Image tensor shape: {img_tensor.shape}, Label: {label}")
    print(f"Class name: {idx_to_class[label]}")