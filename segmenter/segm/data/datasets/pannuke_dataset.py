import os.path as osp
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class PannukeDataset(CustomDataset):
    """Dataset for PanNuke nuclei segmentation.
    
    Args:
        split (str): Split name for PanNuke (training, validation, or testing).
    """
    
    CLASSES = ('background', 'neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial')
    
    PALETTE = [
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Neoplastic - red
        [255, 165, 0],    # Inflammatory - orange
        [0, 255, 0],      # Connective - green
        [139, 69, 19],    # Dead - brown
        [0, 0, 255],      # Epithelial - blue
    ]
    
    def __init__(self, split=None, **kwargs):
        # Map split names to directory names
        self.split_map = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing'
        }
        
        # Store the original split for later use
        self.original_split = split
        
        # Default image and annotation suffixes
        img_suffix = '.png'
        seg_map_suffix = '.png'
        
        # Map the normalized split names to actual directory names
        if split in self.split_map:
            split_dir = self.split_map[split]
        else:
            split_dir = split

        # import pdb; pdb.set_trace()
        
        super(PannukeDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            split=None,  # We'll handle the split manually
            reduce_zero_label=True,
            **kwargs)
        
        # Override the img_dir and ann_dir to include the split directory
        self.img_dir = osp.join(self.img_dir, split_dir)
        self.ann_dir = osp.join(self.ann_dir, split_dir)
        
        # Re-load the image infos with the updated directories
        self.img_infos = self.load_annotations(
            self.img_dir,
            img_suffix,
            self.ann_dir,
            seg_map_suffix,
            self.split
        )
    
    # def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
    #     """Load annotation from directory.
        
    #     Args:
    #         img_dir (str): Path to image directory
    #         img_suffix (str): Suffix of images.
    #         ann_dir (str, optional): Path to annotation directory.
    #         seg_map_suffix (str): Suffix of segmentation maps.
    #         split (str, optional): Split file name. Format: file name without extension.
    #             If split is specified, only file with the name in the split file will be loaded.
        
    #     Returns:
    #         list[dict]: All image info of dataset.
    #     """
    #     import pdb; pdb.set_trace()
    #     img_infos = []
    #     if split is not None:
    #         with open(osp.join(self.data_root, f'{split}.txt')) as f:
    #             for line in f:
    #                 img_name = line.strip()
    #                 img_info = dict(filename=osp.join(img_dir, img_name + img_suffix))
    #                 if ann_dir is not None:
    #                     seg_map = osp.join(ann_dir, img_name + seg_map_suffix)
    #                     img_info['ann'] = dict(seg_map=seg_map)
    #                 img_infos.append(img_info)
    #     else:
    #         # If no split file provided, use the default loading from parent class
    #         return super(PannukeDataset, self).load_annotations(
    #             img_dir, img_suffix, ann_dir, seg_map_suffix)
    #     return img_infos
        
    def get_gt_seg_map_by_idx(self, index):
        """Get ground truth segmentation map by index."""
        seg_map_path = self.img_infos[index]['ann']['seg_map']
        seg_map = np.array(Image.open(seg_map_path))
        return seg_map