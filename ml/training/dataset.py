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

def detect_classes_from_dataset(data_dir):
    """Automatically detect class names from dataset folder structure."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    # Look for folders that contain images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    classes = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            has_images = any(
                child.suffix.lower() in image_extensions 
                for child in item.iterdir() 
                if child.is_file()
            )
            if has_images:
                classes.append(item.name.lower().strip())
    
    if not classes:
        raise ValueError(f"No class folders with images found in {data_dir}")
    
    classes.sort() 
    print(f"Detected {len(classes)} classes from dataset: {classes}")
    return classes

def get_class_mapping(data_dir=None):
    """Get class to index mapping."""
    if data_dir:
        classes = detect_classes_from_dataset(data_dir)
    else:
        # Fallback to hardcoded classes for backward compatibility
        classes = ['barred_owl', 'bear', 'coyote', 'hummingbird', 'osprey', 'sandhill_crane']
        print(f"Using default classes: {classes}")
    
    return {cls: i for i, cls in enumerate(classes)}

def get_idx_to_class(data_dir=None):
    """Get index to class mapping."""
    if data_dir:
        classes = detect_classes_from_dataset(data_dir)
    else:
        # Fallback to hardcoded classes for backward compatibility
        classes = ['barred_owl', 'bear', 'coyote', 'hummingbird', 'osprey', 'sandhill_crane']
        print(f"Using default classes: {classes}")
    
    return {i: cls for i, cls in enumerate(classes)}

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
    from collections import defaultdict
    from sklearn.model_selection import train_test_split
    
    paths_by_class = defaultdict(list)
    
    for image_path in image_paths:
        path_obj = Path(image_path)
        class_name = path_obj.parent.name.lower().strip()
        
        if class_name not in class_mapping:
            class_name = class_name.replace(' ', '_')
        
        if class_name in class_mapping:
            paths_by_class[class_name].append(image_path)
        else:
            print(f"Warning: Skipping image with unknown class: {class_name}")
    
    train_paths = []
    val_paths = []
    test_paths = []
    
    for class_name, class_paths in paths_by_class.items():
        if len(class_paths) < 3:
            print(f"Warning: Class '{class_name}' has only {len(class_paths)} images. Using all for training.")
            train_paths.extend(class_paths)
            continue
        
        # First split: train vs (val+test)
        test_size = 1 - train_ratio  
        train, temp = train_test_split(
            class_paths, 
            test_size=test_size, 
            random_state=42
        )
        
        # Second split: val vs test
        val_size = val_ratio / test_size 
        val, test = train_test_split(
            temp, 
            test_size=val_size, 
            random_state=42
        )
        
        train_paths.extend(train)
        val_paths.extend(val)
        test_paths.extend(test)
    
    print(f"Data split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    return train_paths, val_paths, test_paths

if __name__ == "__main__":
    data_dir = "../photo_data"
    paths = get_image_paths(data_dir)
    mapping = get_class_mapping(data_dir) 
    idx_to_class = get_idx_to_class(data_dir)
    
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