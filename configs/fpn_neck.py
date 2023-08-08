_base_ = ['../configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py']

model=dict(
    decode_head=dict(
        input_transform="resize_concat",
        in_channels=(1024, 1024, 1024, 1024),
        in_index=(0, 1, 2, 3),
    ),
)

data=dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    val=dict(split='ImageSets/SegmentationContext/tiny_val.txt',),
    test=dict(split='ImageSets/SegmentationContext/tiny_val.txt',),
)

optimizer = dict(
    type='AdamW', 
    lr=4e-5, 
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9),
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0, 
    by_epoch=False,
)

checkpoint_config = dict(interval=2000,)
evaluation = dict(interval=10000,)
