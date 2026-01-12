import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    pass

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, max_epochs=15):
    pass

if __name__ == "__main__":
    pass

