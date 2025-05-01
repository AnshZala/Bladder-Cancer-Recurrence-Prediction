import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from matplotlib.colors import ListedColormap

# Import your model and dataset classes from the training script
# Assuming the model definition is available in the same directory
from new_model import ImprovedUNet, ImprovedSegmentationDataset

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def visualize_results(model_path, num_samples=5, save_dir="visualization_results"):
    """
    Visualize the test results of the UNet model.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        num_samples (int): Number of test samples to visualize
        save_dir (str): Directory to save the visualization results
    """
    # Create directory to save visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model
    model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Dice: {checkpoint['dice']:.4f}, IoU: {checkpoint['iou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Model state loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Set paths for dataset - same as in training script
    image_dir = os.path.join("processed_data", "images")
    mask_dir = os.path.join("processed_data", "masks_cleaned")
    
    # Create dataset with no augmentation for testing
    test_dataset = ImprovedSegmentationDataset(image_dir, mask_dir, augment=False)
    
    # Split dataset into train and test
    total_size = len(test_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Use the same seed as during training for reproducibility
    _, test_subset = torch.utils.data.random_split(
        test_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_subset,
        batch_size=1,  # Process one image at a time for visualization
        shuffle=False
    )
    
    # Custom colormap for overlay (red for tumor)
    tumor_cmap = ListedColormap(['none', 'red'])
    
    # Calculate metrics for all test samples
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= num_samples and num_samples > 0:
                break
                
            # Move data to device
            image = image.to(device)
            mask = mask.to(device)
            
            # Get prediction
            output = model(image)
            pred_mask = (output > 0.5).float()
            
            # Calculate metrics
            pred_flat = pred_mask.view(-1).cpu().numpy()
            target_flat = mask.view(-1).cpu().numpy()
            
            # Calculate true positives, false positives, false negatives
            smooth = 1e-7
            tp = np.sum(pred_flat * target_flat)
            fp = np.sum(pred_flat * (1 - target_flat))
            fn = np.sum((1 - pred_flat) * target_flat)
            
            # Calculate Dice coefficient
            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            dice_scores.append(dice)
            
            # Calculate IoU (Jaccard index)
            iou = (tp + smooth) / (tp + fp + fn + smooth)
            iou_scores.append(iou)
            
            # Convert tensors to numpy arrays for visualization
            image_np = image.cpu().squeeze().numpy()
            mask_np = mask.cpu().squeeze().numpy()
            pred_np = pred_mask.cpu().squeeze().numpy()
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot original image
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(image_np, cmap='gray')
            axes[1].imshow(mask_np, cmap=tumor_cmap, alpha=0.3)
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Plot predicted mask
            axes[2].imshow(image_np, cmap='gray')
            axes[2].imshow(pred_np, cmap=tumor_cmap, alpha=0.3)
            axes[2].set_title(f'Predicted Mask\nDice: {dice:.4f}, IoU: {iou:.4f}')
            axes[2].axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f'test_sample_{i}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Processed test sample {i} - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    # Print average metrics
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"\nAverage Test Metrics - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
    
    # Create additional visualization showing the best and worst predictions
    if len(dice_scores) >= 2:
        best_idx = np.argmax(dice_scores)
        worst_idx = np.argmin(dice_scores)
        
        print(f"Best prediction: Sample {best_idx} - Dice: {dice_scores[best_idx]:.4f}")
        print(f"Worst prediction: Sample {worst_idx} - Dice: {dice_scores[worst_idx]:.4f}")

def generate_overlay_outputs(model_path, output_dir="overlay_results"):
    """
    Generate overlaid images showing the tumor segmentation on original images.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        output_dir (str): Directory to save the overlaid images
    """
    # Create directory to save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Set paths for dataset
    image_dir = os.path.join("processed_data", "images")
    mask_dir = os.path.join("processed_data", "masks_cleaned")
    
    # Create dataset
    dataset = ImprovedSegmentationDataset(image_dir, mask_dir, augment=False)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    _, test_subset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloader
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            # Get file name from the dataset
            file_idx = test_subset.indices[i]
            file_name = dataset.image_files[file_idx]
            
            # Process with model
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            pred_mask = (output > 0.5).float()
            
            # Convert to numpy
            image_np = image.cpu().squeeze().numpy() * 255  # Scale back to 0-255
            image_np = image_np.astype(np.uint8)
            pred_np = pred_mask.cpu().squeeze().numpy() * 255
            pred_np = pred_np.astype(np.uint8)
            mask_np = mask.cpu().squeeze().numpy() * 255
            mask_np = mask_np.astype(np.uint8)
            
            # Create colored mask for overlay
            # Convert grayscale to BGR for colored overlay
            image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
            # Create overlay mask (red for predicted tumor)
            overlay_pred = image_color.copy()
            overlay_pred[pred_np > 127, 2] = 255  # Red channel
            
            # Create overlay for ground truth (green for ground truth tumor)
            overlay_gt = image_color.copy()
            overlay_gt[mask_np > 127, 1] = 255  # Green channel
            
            # Create combined visualization (blue for both prediction and ground truth)
            overlay_combined = image_color.copy()
            overlay_combined[pred_np > 127, 2] = 255  # Red for prediction
            overlay_combined[mask_np > 127, 1] = 255  # Green for ground truth
            # Where both overlap, it will appear yellowish
            
            # Add alpha blending
            alpha = 0.6
            overlay_pred = cv2.addWeighted(image_color, 1-alpha, overlay_pred, alpha, 0)
            overlay_gt = cv2.addWeighted(image_color, 1-alpha, overlay_gt, alpha, 0)
            overlay_combined = cv2.addWeighted(image_color, 1-alpha, overlay_combined, alpha, 0)
            
            # Save images
            cv2.imwrite(os.path.join(output_dir, f"{file_name}_original.png"), image_np)
            cv2.imwrite(os.path.join(output_dir, f"{file_name}_prediction.png"), overlay_pred)
            cv2.imwrite(os.path.join(output_dir, f"{file_name}_ground_truth.png"), overlay_gt)
            cv2.imwrite(os.path.join(output_dir, f"{file_name}_combined.png"), overlay_combined)
            
            print(f"Processed image {i+1}/{len(test_loader)}: {file_name}")

if __name__ == "__main__":
    # Set the path to your best model checkpoint
    model_path = os.path.join('checkpoints', 'only_tumor_seg.pth')
    
    # Visualize test results
    print("Generating visualization plots...")
    visualize_results(model_path, num_samples=10)  # Visualize 10 test samples
    
    # Generate overlay images
    print("\nGenerating overlay images...")
    generate_overlay_outputs(model_path)
    
    print("\nVisualization complete. Check the output directories for results.")