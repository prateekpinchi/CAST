# dataset settings
dataset_type = "PannukeDataset"
data_root = "data/PanNuke"
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True
)
crop_size = (224, 224)
max_ratio = 2
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=(224, 224), keep_ratio=False),
    dict(type="RandomFlip", prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(256 * max_ratio, 256),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(256 * max_ratio, 256),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="annotations",
        split="train",  # Will be mapped to "training" in the dataset class
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="annotations",
        split="val",    # Will be mapped to "validation" in the dataset class
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="annotations",
        split="test",   # Will be mapped to "testing" in the dataset class
        pipeline=test_pipeline,
    ),
)