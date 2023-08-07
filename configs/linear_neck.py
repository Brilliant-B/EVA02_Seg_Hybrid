_base_ = ['../configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py']

model=dict(
    decode_head=dict(),
)

data=dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val=dict(split='ImageSets/SegmentationContext/tiny_val.txt',),
    test=dict(split='ImageSets/SegmentationContext/tiny_val.txt',),
)

checkpoint_config = dict(interval=2000,)
evaluation = dict(interval=10000,)
