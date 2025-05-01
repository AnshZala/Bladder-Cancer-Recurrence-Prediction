import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import radiomics
from radiomics import featureextractor

# Configure radiomics feature extractor with correct 2D settings
settings = {}
settings['binWidth'] = 25  # Bin width for discretization
settings['force2D'] = True  # Force 2D extraction
settings['resampledPixelSpacing'] = None  # Don't resample - use original pixel spacing
settings['interpolator'] = sitk.sitkBSpline
settings['correctMask'] = True
settings['normalize'] = True
settings['normalizeScale'] = 100

# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllFeatures()  # Enable all feature classes

def is_mask_non_empty(mask_path):
    """Check if mask contains any non-zero pixels"""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return np.any(mask > 0)
    except Exception as e:
        print(f"Error checking mask {mask_path}: {e}")
        return False

def extract_features_from_slice(image_path, mask_path, patient_id, slice_id):
    """Extract radiomics features from a single image-mask pair"""
    try:
        # Read image and mask using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read image at {image_path}")
            return None
            
        if mask is None:
            print(f"Warning: Could not read mask at {mask_path}")
            return None
        
        # Ensure mask is binary (0 and 1 values only)
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        # Convert to SimpleITK images - explicitly as 2D
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)
        
        # Extract features
        features = extractor.execute(sitk_image, sitk_mask)
        
        # Create a dictionary with patient information and features
        result = {
            'patient_id': patient_id,
            'slice_id': slice_id,
            'image_path': image_path,
            'mask_path': mask_path
        }
        
        # Add all computed features to the result dictionary
        for feature_name, feature_value in features.items():
            # Skip the diagnostics information
            if feature_name.startswith('diagnostics_'):
                continue
            # Convert numpy types to Python native types for CSV compatibility
            if isinstance(feature_value, (np.float32, np.float64)):
                result[feature_name] = float(feature_value)
            elif isinstance(feature_value, (np.int32, np.int64)):
                result[feature_name] = int(feature_value)
            else:
                result[feature_name] = feature_value
                
        return result
    
    except Exception as e:
        print(f"Error extracting features from {image_path} and {mask_path}: {e}")
        return None

def find_corresponding_image(mask_path, image_directory):
    """Find corresponding image file for a given mask"""
    # Extract patient ID and slice info from mask path
    # This is based on your specific filename format
    mask_filename = os.path.basename(mask_path)
    
    # Extract patient ID from the mask path
    # Example: "TCGA_segmentation_results/masks/TCGA-ZF-A9R3/07-30-2002-NA-Bladder CT-07985/2.000000-AXIAL-6949/1-002.dcm_mask.png"
    path_parts = mask_path.split(os.sep)
    patient_id = None
    for part in path_parts:
        if part.startswith("TCGA-ZF-"):
            patient_id = part
            break
    
    if not patient_id:
        print(f"Could not determine patient ID from mask path: {mask_path}")
        return None
        
    # Extract slice identifier from mask filename
    slice_id = mask_filename.split('.dcm_mask')[0]  # e.g., "1-002"
    
    # Try to find the corresponding image in the image directory
    # Pattern: looking for a similar path structure but without the "_mask" part
    potential_image_paths = glob.glob(os.path.join(image_directory, "**", f"*{slice_id}*.dcm.png"), recursive=True)
    
    if potential_image_paths:
        # Filter for paths containing the patient ID
        matching_paths = [p for p in potential_image_paths if patient_id in p]
        if matching_paths:
            return matching_paths[0]
        return potential_image_paths[0]  # If no exact match, take the first potential match
    
    return None

def process_patient_data(image_directory, mask_directory, output_csv_path):
    """Process all patients and their slices to extract features"""
    all_features = []
    total_masks_processed = 0
    
    # Get all patient directories in the mask folder
    patient_dirs = [d for d in os.listdir(mask_directory) if os.path.isdir(os.path.join(mask_directory, d))]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    for patient_id in tqdm(patient_dirs, desc="Processing patients"):
        patient_mask_dir = os.path.join(mask_directory, patient_id)
        
        # Get all mask files for this patient (recursively search subdirectories)
        mask_files = glob.glob(os.path.join(patient_mask_dir, '**', '*.png'), recursive=True)
        
        print(f"Found {len(mask_files)} mask files for patient {patient_id}")
        
        patient_features = []
        
        for mask_path in tqdm(mask_files, desc=f"Processing {patient_id}", leave=False):
            # Check if mask is non-empty
            if is_mask_non_empty(mask_path):
                # Find corresponding image
                image_path = find_corresponding_image(mask_path, image_directory)
                
                if image_path and os.path.exists(image_path):
                    # Extract slice ID from the filename
                    mask_filename = os.path.basename(mask_path)
                    slice_id = mask_filename.split('.dcm_mask')[0]  # Adjust based on your actual naming pattern
                    
                    # Extract features
                    features = extract_features_from_slice(image_path, mask_path, patient_id, slice_id)
                    
                    if features:
                        patient_features.append(features)
                        total_masks_processed += 1
                else:
                    print(f"Warning: Could not find corresponding image for mask {mask_path}")
            else:
                print(f"Skipping empty mask: {mask_path}")
        
        # Add all features from this patient
        all_features.extend(patient_features)
        print(f"Extracted features from {len(patient_features)} non-empty masks for patient {patient_id}")
    
    # Create DataFrame and save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully extracted features for {total_masks_processed} slices across {len(patient_dirs)} patients.")
        print(f"Results saved to {output_csv_path}")
    else:
        print("No features were extracted. Check your mask files and paths.")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual directories
    image_directory = "Subset_TCGA"
    mask_directory = "TCGA_segmentation_results/masks"
    output_csv_path = "radiomics_features.csv"
    
    process_patient_data(image_directory, mask_directory, output_csv_path)