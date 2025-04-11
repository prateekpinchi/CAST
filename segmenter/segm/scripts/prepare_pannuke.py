"""Prepare PanNuke dataset"""
import click
import os
from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def save_semantic_mask_as_rgb(mask, output_path=None, class_colors=None):
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
    if output_path is not None:
        Image.fromarray(rgb_mask).save(output_path)
    
    return rgb_mask, class_colors  # Return the color mapping for reference

def download_and_process_pannuke(output_path, overwrite=False, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Download PanNuke dataset and convert to expected format"""
    # Create output directories
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create directories for train/val/test splits
    for split in ["training", "validation", "testing"]:
        (output_path / f"images/{split}").mkdir(exist_ok=True, parents=True)
        (output_path / f"annotations/{split}").mkdir(exist_ok=True, parents=True)
    
    # Load dataset with force_download to avoid caching issues
    print("Downloading PanNuke dataset...")
    dataset_root = '/home/pinchi/cast4nuseg/hierseg/test_dataset'

    class_colors = {
        1: (255, 0, 0),    # Red: Neoplastic
        2: (255, 165, 0),  # Orange: Inflammatory
        3: (0, 255, 0),    # Green: Connective
        4: (139, 69, 19),    # Brown: Dead
        5: (0, 0, 255),    # Blue: Epithelial
    }

    dfs = []
    for idx, split in enumerate(["training", "validation", "testing"]):
        df = pd.read_csv(os.path.join(dataset_root, str(idx), "data.csv"))
        dfs.append(df)

    ds_df = pd.concat(dfs)

    imgs = []
    masks = []
    for img_path, group in tqdm(ds_df.groupby("image")):
        # Read in original image
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        imgs.append(img)

        # Create empty semantic mask
        semantic_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for i, mask_path in zip(group['Unnamed: 0'], group['mask']):
            class_label = group['category'][i] + 1  # Add 1 to make background 0
            mask = Image.open(mask_path)
            mask = np.array(mask)
            semantic_mask[mask > 0] = class_label

        mask, _ = save_semantic_mask_as_rgb(semantic_mask, class_colors=class_colors)
        masks.append(mask)
    num_samples = len(imgs)
    assert num_samples == len(masks), "Number of images and masks do not match"

    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    splits = {
        "training": {
            'imgs': imgs[:num_train],
            'masks': masks[:num_train]
        },
        "validation": {
            'imgs': imgs[num_train:num_train + num_val],
            'masks': masks[num_train:num_train + num_val]
        },
        "testing": {
            'imgs': imgs[num_train + num_val:],
            'masks': masks[num_train + num_val:]
        }
    }

    for split, info in splits.items():
        imgs = info['imgs']
        masks = info['masks']
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            # Save images and masks
            img_filename = output_path / f"images/{split}/img_{i:04}.png"
            mask_filename = output_path / f"annotations/{split}/img_{i:04}.png"
            if not img_filename.exists() or overwrite:
                Image.fromarray(img).save(img_filename)
            if not mask_filename.exists() or overwrite:
                Image.fromarray(mask).save(mask_filename)

    print("Dataset preparation complete!")
    return True


@click.command(help="Initialize PanNuke dataset.")
@click.argument("download_dir", type=str)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@click.option("--train_ratio", default=0.8, type=float, help="Ratio of training data")
@click.option("--val_ratio", default=0.1, type=float, help="Ratio of validation data")
@click.option("--test_ratio", default=0.1, type=float, help="Ratio of test data")
def main(download_dir, overwrite, train_ratio, val_ratio, test_ratio):
    """
    Main function to prepare the PanNuke dataset.
    Args:
        download_dir: Directory to save the dataset
        overwrite: Overwrite existing files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    dataset_dir = Path(download_dir) / "PanNuke"
    success = download_and_process_pannuke(dataset_dir, overwrite, train_ratio, val_ratio, test_ratio)
    if success:
        print(f"PanNuke dataset prepared at {dataset_dir}")
    else:
        print("Failed to prepare PanNuke dataset")

if __name__ == "__main__":
    main()