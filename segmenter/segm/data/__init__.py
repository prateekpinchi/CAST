from segm.data.loader import Loader

from segm.data.imagenet import ImagenetDataset
from segm.data.ade20k import ADE20KSegmentation
from segm.data.pascal_context import PascalContextDataset
from segm.data.cityscapes import CityscapesDataset
from segm.data.pannuke import PannukeSegmentation

# Import datasets to register them with MMSegmentation
from segm.data.datasets import PannukeDataset
