_base_ = ['../configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py']

model=dict(
    decode_head=dict(
        multi_scale_adapter=True,
        input_transform="resize_concat",
        in_channels=(256, 512, 1024, 1024),
        in_index=(0, 1, 2, 3),
    ),
)

data=dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    val=dict(split='ImageSets/SegmentationContext/tiny_val.txt',),
    test=dict(split='ImageSets/SegmentationContext/val.txt',),
)

checkpoint_config = dict(interval=2000,)
evaluation = dict(interval=10000,)
