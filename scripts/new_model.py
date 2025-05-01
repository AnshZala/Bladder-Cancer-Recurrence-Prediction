import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Check if CUDA is available - only in main process
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved Double Convolution Block with Batch Normalization
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Improved U-Net with Deeper Architecture
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Middle part of U-Net
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse the skip connections list for easier access
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle cases where dimensions don't match exactly
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # Final convolution and sigmoid for binary segmentation
        return self.sigmoid(self.final_conv(x))

# Data augmentation class
class Augmentation:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, brightness_contrast_prob=0.3):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_contrast_prob = brightness_contrast_prob
    
    def __call__(self, image, mask):
        # Horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Vertical flip
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Rotation
        if random.random() < self.rotate_prob:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Brightness and contrast adjustment (only for image, not mask)
        if random.random() < self.brightness_contrast_prob:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, mask

# Improved segmentation dataset with augmentation
class ImprovedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.augmentation = Augmentation()
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Normalize
        image = image / 255.0
        mask = mask / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        
        # Apply augmentation if needed
        if self.augment:
            image, mask = self.augmentation(image, mask)
            
        return image, mask

# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection
        intersection = (pred_flat * target_flat).sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return Dice Loss
        return 1.0 - dice

# Combined loss function: BCE + Dice
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        combined_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss
        return combined_loss

# Helper function to calculate Dice coefficient and IoU for validation
def calculate_metrics(pred, target, threshold=0.5):
    # Apply threshold to predictions
    pred = (pred > threshold).float()
    
    # Flatten predictions and targets
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum(pred_flat * target_flat)
    fp = np.sum(pred_flat * (1 - target_flat))
    fn = np.sum((1 - pred_flat) * target_flat)
    
    # Calculate Dice coefficient
    smooth = 1e-7
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # Calculate IoU (Jaccard index)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    return dice, iou

# Training function
def train_model():
    # Set paths and ensure correct path format
    image_dir = os.path.join("processed_data", "images")
    mask_dir = os.path.join("processed_data", "masks_cleaned")
    
    # Verify directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: Directories not found. Please check paths:")
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        return
    
    # Create datasets with augmentation for training
    train_dataset = ImprovedSegmentationDataset(image_dir, mask_dir, augment=True)
    val_dataset = ImprovedSegmentationDataset(image_dir, mask_dir, augment=False)
    
    # Check if dataset is empty
    if len(train_dataset) == 0:
        print("Error: No image or mask files found")
        return
    
    print(f"Found {len(train_dataset)} image-mask pairs")
    
    # Split dataset into train and validation
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create dataloaders with reduced workers to prevent excessive forking
    train_loader = DataLoader(
        train_subset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True  # Enable faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True  # Enable faster data transfer to GPU
    )
    
    # Initialize model, loss, and optimizer
    model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss(weight_bce=0.3, weight_dice=0.7)  # More weight on Dice loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, 
        verbose=True, min_lr=1e-6
    )
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    num_epochs = 30
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    counter = 0
    
    # Create folder for model checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, masks in train_loader:
            batch_count += 1
            # Move data to GPU
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        dice_scores = []
        iou_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
                # Calculate metrics
                batch_dice, batch_iou = calculate_metrics(outputs, masks)
                dice_scores.append(batch_dice)
                iou_scores.append(batch_iou)
        
        val_loss = val_loss / len(val_loader.dataset)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Metrics: Average Dice coefficient: {avg_dice:.4f} Average IoU (Jaccard index): {avg_iou:.4f}')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset early stopping counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dice': avg_dice,
                'iou': avg_iou
            }, os.path.join('checkpoints', 'unet_model_best.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            counter += 1  # Increment early stopping counter
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'dice': avg_dice,
                'iou': avg_iou
            }, os.path.join('checkpoints', f'unet_model_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join('checkpoints', 'unet_model_final.pth'))
    print("Training completed. Final model saved as 'unet_model_final.pth'")

if __name__ == "__main__":
    train_model()