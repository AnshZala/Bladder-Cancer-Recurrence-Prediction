import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import shutil
from torch.utils.data import Dataset, DataLoader

# Import your model definition from the original script
from new_model import ImprovedUNet

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class for TCGA-BLCA images (without ground truth masks)
class TCGAImageDataset(Dataset):
    def __init__(self, image_dir, size=256):
        self.image_dir = image_dir
        self.size = size
        
        # Get all PNG files recursively
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.dcm'):
                    self.image_files.append(os.path.join(root, file))
        
        self.image_files.sort()  # Sort files for consistent ordering
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.size, self.size))
        
        # Normalize
        image = image / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        
        return image, img_path

def segment_tcga_images(model_path, tcga_dir, output_dir="tcga_segmentation_results"):
    """
    Apply the trained UNet model to segment tumor regions in TCGA-BLCA images.
    
    Args:
        model_path (str): Path to the trained model checkpoint
        tcga_dir (str): Directory containing TCGA-BLCA images
        output_dir (str): Directory to save the segmentation results
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    
    # Load the model
    model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']} with metrics:")
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"Dice: {checkpoint['dice']:.4f}, IoU: {checkpoint['iou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Model state loaded successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataset for TCGA images
    tcga_dataset = TCGAImageDataset(tcga_dir)
    
    if len(tcga_dataset) == 0:
        print(f"No images found in {tcga_dir}")
        return
    
    print(f"Found {len(tcga_dataset)} images to process")
    
    # Create dataloader
    tcga_loader = DataLoader(
        tcga_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=2
    )
    
    # Process images and save results
    with torch.no_grad():
        for images, img_paths in tqdm(tcga_loader, desc="Processing TCGA images"):
            # Get image path as string
            img_path = img_paths[0]
            
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Create subfolder structure matching original path
            relative_path = os.path.relpath(os.path.dirname(img_path), tcga_dir)
            if relative_path == '.':
                relative_path = ''
            
            output_subdir = os.path.join(output_dir, "masks", relative_path)
            overlay_subdir = os.path.join(output_dir, "overlays", relative_path)
            
            os.makedirs(output_subdir, exist_ok=True)
            os.makedirs(overlay_subdir, exist_ok=True)
            
            # Move data to device
            images = images.to(device)
            
            # Get prediction
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()
            
            # Convert to numpy
            image_np = images.cpu().squeeze().numpy() * 255
            image_np = image_np.astype(np.uint8)
            pred_np = pred_masks.cpu().squeeze().numpy() * 255
            pred_np = pred_np.astype(np.uint8)
            
            # Save the prediction mask
            mask_path = os.path.join(output_subdir, f"{filename}_mask.png")
            cv2.imwrite(mask_path, pred_np)
            
            # Create colored mask for overlay
            image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
            # Create overlay mask (red for tumor)
            overlay = image_color.copy()
            overlay[pred_np > 127, 2] = 255  # Red channel
            
            # Add alpha blending
            alpha = 0.6
            overlay_result = cv2.addWeighted(image_color, 1-alpha, overlay, alpha, 0)
            
            # Save the overlay
            overlay_path = os.path.join(overlay_subdir, f"{filename}_overlay.png")
            cv2.imwrite(overlay_path, overlay_result)
            
            # Create side-by-side visualization for easier comparison
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original image
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot image with segmentation overlay
            axes[1].imshow(image_np, cmap='gray')
            mask_display = np.zeros_like(image_np)
            mask_display[pred_np > 127] = 1
            axes[1].imshow(mask_display, cmap='hot', alpha=0.4)
            axes[1].set_title('Tumor Segmentation')
            axes[1].axis('off')
            
            # Adjust layout and save
            plt.tight_layout()
            visualization_path = os.path.join(overlay_subdir, f"{filename}_visualization.png")
            plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\nSegmentation complete. Results saved to {output_dir}")
    
    # Create a summary HTML report
    create_html_report(output_dir, tcga_dataset.image_files)

def create_html_report(output_dir, image_files):
    """
    Create an HTML report summarizing the segmentation results.
    
    Args:
        output_dir (str): Directory containing segmentation results
        image_files (list): List of processed image files
    """
    html_path = os.path.join(output_dir, "segmentation_report.html")
    
    with open(html_path, 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>TCGA-BLCA Tumor Segmentation Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333366; }
                .result-row { display: flex; margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }
                .result-item { margin-right: 20px; }
                img { max-width: 300px; border: 1px solid #ccc; }
                .filename { font-weight: bold; margin-bottom: 5px; }
            </style>
        </head>
        <body>
            <h1>TCGA-BLCA Tumor Segmentation Results</h1>
            <p>Segmentation performed using UNet model</p>
            <div id="results">
        ''')
        
        # Add a section for each image
        for img_path in image_files[:50]:  # Limit to first 50 images to avoid huge HTML file
            filename = os.path.splitext(os.path.basename(img_path))[0]
            relative_path = os.path.relpath(os.path.dirname(img_path), output_dir)
            
            # Get paths to result images
            mask_path = os.path.join("masks", relative_path, f"{filename}_mask.png")
            overlay_path = os.path.join("overlays", relative_path, f"{filename}_overlay.png")
            viz_path = os.path.join("overlays", relative_path, f"{filename}_visualization.png")
            
            f.write(f'''
            <div class="result-row">
                <div class="result-item">
                    <div class="filename">Original: {filename}</div>
                    <img src="{os.path.relpath(img_path, output_dir)}" alt="Original Image">
                </div>
                <div class="result-item">
                    <div class="filename">Segmentation Mask</div>
                    <img src="{mask_path}" alt="Segmentation Mask">
                </div>
                <div class="result-item">
                    <div class="filename">Overlay</div>
                    <img src="{overlay_path}" alt="Overlay Image">
                </div>
            </div>
            ''')
        
        if len(image_files) > 50:
            f.write(f'<p>Showing 50 of {len(image_files)} total images. See output directory for all results.</p>')
            
        f.write('''
            </div>
        </body>
        </html>
        ''')
    
    print(f"HTML report created: {html_path}")

def analyze_tumor_statistics(output_dir):
    """
    Analyze tumor statistics from the generated masks.
    
    Args:
        output_dir (str): Directory containing segmentation results
    """
    masks_dir = os.path.join(output_dir, "masks")
    mask_files = []
    
    # Get all mask files
    for root, _, files in os.walk(masks_dir):
        for file in files:
            if file.endswith('_mask.png'):
                mask_files.append(os.path.join(root, file))
    
    if not mask_files:
        print("No mask files found to analyze")
        return
    
    print(f"Analyzing {len(mask_files)} tumor masks...")
    
    # Collect tumor statistics
    tumor_areas = []
    tumor_percentages = []
    has_tumor = 0
    
    for mask_path in tqdm(mask_files, desc="Analyzing tumor statistics"):
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Count tumor pixels (values > 127)
        tumor_pixels = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        # Record statistics
        if tumor_pixels > 0:
            has_tumor += 1
            tumor_areas.append(tumor_pixels)
            tumor_percentages.append(tumor_percentage)
    
    # Generate statistics report
    stats_path = os.path.join(output_dir, "tumor_statistics.txt")
    
    with open(stats_path, 'w') as f:
        f.write("TCGA-BLCA Tumor Segmentation Statistics\n")
        f.write("======================================\n\n")
        f.write(f"Total images analyzed: {len(mask_files)}\n")
        f.write(f"Images with detected tumor: {has_tumor} ({has_tumor/len(mask_files)*100:.2f}%)\n\n")
        
        if tumor_areas:
            f.write("Tumor Area Statistics (in pixels):\n")
            f.write(f"  Mean area: {np.mean(tumor_areas):.2f}\n")
            f.write(f"  Median area: {np.median(tumor_areas):.2f}\n")
            f.write(f"  Minimum area: {np.min(tumor_areas):.2f}\n")
            f.write(f"  Maximum area: {np.max(tumor_areas):.2f}\n")
            f.write(f"  Standard deviation: {np.std(tumor_areas):.2f}\n\n")
            
            f.write("Tumor Percentage Statistics (% of image):\n")
            f.write(f"  Mean percentage: {np.mean(tumor_percentages):.2f}%\n")
            f.write(f"  Median percentage: {np.median(tumor_percentages):.2f}%\n")
            f.write(f"  Minimum percentage: {np.min(tumor_percentages):.2f}%\n")
            f.write(f"  Maximum percentage: {np.max(tumor_percentages):.2f}%\n")
            f.write(f"  Standard deviation: {np.std(tumor_percentages):.2f}%\n")
    
    print(f"Tumor statistics saved to {stats_path}")
    
    # Create histogram of tumor percentages
    if tumor_percentages:
        plt.figure(figsize=(10, 6))
        plt.hist(tumor_percentages, bins=20, color='teal', alpha=0.7)
        plt.xlabel('Tumor Percentage (%)')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Tumor Size in TCGA-BLCA Dataset')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        hist_path = os.path.join(output_dir, "tumor_size_distribution.png")
        plt.savefig(hist_path, dpi=200)
        plt.close()
        
        print(f"Tumor size distribution histogram saved to {hist_path}")

if __name__ == "__main__":
    # Paths 
    model_path = os.path.join('checkpoints', 'only_tumor_seg.pth')  # Update if needed
    
    # Directory containing TCGA-BLCA images with the structure as shown in your screenshot
    tcga_dir = os.path.join("TCGA-ZF-AA5P")
    
    # Directory to save segmentation results
    output_dir = os.path.join("TCGA_segmentation_results")
    
    # Segment TCGA images
    segment_tcga_images(model_path, tcga_dir, output_dir)
    
    # Analyze tumor statistics
    analyze_tumor_statistics(output_dir)
    
    print("TCGA-BLCA dataset segmentation and analysis complete!")