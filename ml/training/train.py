import torch
import torch.nn as nn
import json
import re
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_next_version(model_registry_dir):
    registry_path = Path(model_registry_dir)
    registry_path.mkdir(exist_ok=True)
    
    # Find existing models
    existing_models = list(registry_path.glob("model_v*.pth"))

    if not existing_models:
        return 1

    versions = []
    for model_file in existing_models:
        match = re.search(r'model_v(\d+)\.pth', model_file.name)
        if match:
            versions.append(int(match.group(1)))
    
    if not versions:
        return 1
    
    return max(versions) + 1

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def save_model_with_metadata(model, model_registry_dir, training_metrics, class_mapping, idx_to_class):
    version = get_next_version(model_registry_dir)
    
    # Save model
    model_path = f"{model_registry_dir}/model_v{version}.pth"
    save_model(model, model_path)
    
    # Create metadata
    metadata = {
        "version": f"v{version}",
        "classes": list(idx_to_class.values()),
        "class_to_idx": class_mapping,
        "training_metrics": training_metrics,
        "preprocessing": {
            "input_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "model_architecture": "MobileNetV3",
        "trained_date": datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = f"{model_registry_dir}/model_v{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model and metadata saved as version {version}")
    return version

def setup_training(model, learning_rate=0.001):
    """Set up loss function, optimizer, and scheduler."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    return criterion, optimizer, scheduler

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train() # Set model to training mode
    total_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass (to get predictions)
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass (calculate gradients)
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def validate(model, val_loader, criterion):
    model.eval()  
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, max_epochs=15):
    best_val_loss = float('inf')
    patience = 0
    max_patience = 3
    
    for epoch in tqdm(range(max_epochs), desc="Training"):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience = 0
            
            # Save best model (Module 5)
            if class_mapping and idx_to_class:
                training_metrics = {
                    "best_val_accuracy": val_acc,
                    "best_val_loss": val_loss,
                    "final_train_loss": train_loss,
                    "epochs_trained": epoch + 1
                }
                
                save_model_with_metadata(
                    model, 
                    model_registry_dir,
                    training_metrics,
                    class_mapping,
                    idx_to_class
                )
            else:
                print(f"  ✓ New best model! (Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%)")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    return model

if __name__ == "__main__":
    print("Training module ready!")

