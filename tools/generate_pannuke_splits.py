import os
import os.path as osp
import random
import argparse
import glob


def generate_split_files(data_root, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Generate train.txt, val.txt and test.txt files for PanNuke dataset.
    
    Args:
        data_root (str): Root directory of PanNuke dataset, containing training, validation, and testing folders
        output_dir (str): Directory to save the split files
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of testing data
        seed (int): Random seed for reproducibility
    """
    # Verify the ratios sum up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum up to 1"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all image files in the dataset
    training_dir = osp.join(data_root, 'images/training')
    validation_dir = osp.join(data_root, 'images/validation')
    testing_dir = osp.join(data_root, 'images/testing')
    
    # Get all the image files (without extension)
    training_files = [osp.splitext(osp.basename(f))[0] for f in glob.glob(osp.join(training_dir, '*.png'))]
    validation_files = [osp.splitext(osp.basename(f))[0] for f in glob.glob(osp.join(validation_dir, '*.png'))]
    testing_files = [osp.splitext(osp.basename(f))[0] for f in glob.glob(osp.join(testing_dir, '*.png'))]
    
    # Combine all files and shuffle
    # all_files = training_files + validation_files + testing_files
    # random.shuffle(all_files)
    
    # Calculate the number of samples for each split
    # num_samples = len(all_files)
    # num_train = int(num_samples * train_ratio)
    # num_val = int(num_samples * val_ratio)
    # num_test = num_samples - num_train - num_val
    
    # Split the files
    # train_files = all_files[:num_train]
    # val_files = all_files[num_train:num_train + num_val]
    # test_files = all_files[num_train + num_val:]
    train_files = training_files
    val_files = validation_files
    test_files = testing_files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the split files
    with open(osp.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(osp.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
    
    with open(osp.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"Split files created successfully in {output_dir}")
    print(f"Training: {len(train_files)} samples")
    print(f"Validation: {len(val_files)} samples")
    print(f"Testing: {len(test_files)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split files for PanNuke dataset")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of PanNuke dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save split files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of testing data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    import pdb; pdb.set_trace()
    
    args = parser.parse_args()
    generate_split_files(args.data_root, args.output_dir, 
                         args.train_ratio, args.val_ratio, args.test_ratio, 
                         args.seed)
