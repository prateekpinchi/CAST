"""Prepare PanNuke dataset"""
import click
import os
from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def save_semantic_mask_as_rgb(mask, output_path, class_colors=None):
    """
    Save a semantic mask as an RGB image with different colors for each class.
    
    Args:
        mask: The semantic mask with class indices
        output_path: Path to save the colorized mask
        class_colors: Optional dictionary mapping class indices to RGB colors
                     If None, it will generate colors automatically
    """
    # Get unique class indices (excluding 0, which is background)
    unique_classes = np.unique(mask)
    if 0 in unique_classes:
        unique_classes = unique_classes[unique_classes != 0]
    
    # Create color mapping if not provided
    if class_colors is None:
        class_colors = {}
        # Generate distinct colors for each class
        import colorsys
        N = len(unique_classes)
        for i, class_idx in enumerate(unique_classes):
            # Generate evenly spaced colors in HSV and convert to RGB
            hue = i / N
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            class_colors[class_idx] = (int(r * 255), int(g * 255), int(b * 255))
    
    # Create RGB image with black background
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Fill each class with its color
    for class_idx, color in class_colors.items():
        rgb_mask[mask == class_idx] = color
    
    # Save the image
    Image.fromarray(rgb_mask).save(output_path)
    
    return class_colors  # Return the color mapping for reference

def download_and_process_pannuke(output_path, overwrite=False):
    """Download PanNuke dataset and convert to expected format"""
    # Create output directories
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create directories for train/val/test splits
    for split in ["training", "validation", "testing"]:
        (output_path / f"images/{split}").mkdir(exist_ok=True, parents=True)
        (output_path / f"annotations/{split}").mkdir(exist_ok=True, parents=True)
    
    # Load dataset with force_download to avoid caching issues
    # import pdb; pdb.set_trace()
    print("Downloading PanNuke dataset...")
    dataset_root = '/home/pinchi/cast4nuseg/hierseg/test_dataset'

    class_colors = {
        1: (255, 0, 0),    # Red: Neoplastic
        2: (255, 165, 0),  # Orange: Inflammatory
        3: (0, 255, 0),    # Green: Connective
        4: (139, 69, 19),    # Brown: Dead
        5: (0, 0, 255),    # Blue: Epithelial
    }

    for idx, split in enumerate(["training", "validation", "testing"]):
        df = pd.read_csv(os.path.join(dataset_root, str(idx), "data.csv"))
        print(f"Processing {split} split...")
        for img_path, group in tqdm(df.groupby("image")):
            # Read in original image
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img)

            # Create empty semantic mask
            semantic_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            for i, mask_path in zip(group['Unnamed: 0'], group['mask']):
                class_label = group['category'][i] + 1  # Add 1 to make background 0
                mask = Image.open(mask_path)
                mask = np.array(mask)
                semantic_mask[mask > 0] = class_label
            
            img_filename = output_path / f"images/{split}/{img_path.split('/')[-1]}"
            mask_filename = output_path / f"annotations/{split}/{img_path.split('/')[-1]}"

            if not img_filename.exists() or overwrite:
                Image.fromarray(img).save(img_filename)
            if not mask_filename.exists() or overwrite:
                save_semantic_mask_as_rgb(semantic_mask, mask_filename, class_colors=class_colors)

    print("Dataset preparation complete!")
    return True


@click.command(help="Initialize PanNuke dataset.")
@click.argument("download_dir", type=str)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(download_dir, overwrite):
    dataset_dir = Path(download_dir) / "PanNuke"
    success = download_and_process_pannuke(dataset_dir, overwrite)
    if success:
        print(f"PanNuke dataset prepared at {dataset_dir}")
    else:
        print("Failed to prepare PanNuke dataset")

if __name__ == "__main__":
    main()