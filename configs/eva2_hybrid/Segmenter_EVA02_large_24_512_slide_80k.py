# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
_base_ = [
    '../../configs/_base_/models/segmenter.py', '../../configs/_base_/datasets/pascal_context.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_80k.py'
]
img_scale = (520, 520)
crop_size = (224, 224)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='EVA2',
        img_size=224, # 重要！
        patch_size=16, 
        in_chans=3,
        embed_dim=1024, # 重要！
        depth=24,
        num_heads=16, 
        mlp_ratio=4*2/3,      # GLU default
        out_indices=[7, 11, 15, 23],
        qkv_bias=True, 
        drop_rate=0.1, 
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        init_values=None, 
        use_checkpoint=False, 
        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        subln=True,
        xattn=False,
        naiveswiglu=True,
        pretrained='pretrained/eva02_L_pt_m38m_p14to16.pt',
    ),
    decode_head=dict(
        type='SegmenterHead_maskT',
        img_size_ori=224, # 重要！
        n_layers=2,
        n_heads=16,
        d_model=1024, # 重要！
        d_ff=4*1024,
        drop_path_rate=0.0,
        dropout=0.1,
        multi_scale_adapter=False,
        input_transform=None,
        in_channels=1024,
        in_index=-1,
        channels=1024,
        dropout_ratio=0, # no relation
        num_classes=60,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(150, 150))
)

optimizer = dict(_delete_=True, 
                 type='AdamW', 
                 lr=4e-5, 
                 betas=(0.9, 0.999), 
                 weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
)

checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=10)
evaluation = dict(interval=10000, metric='mIoU', save_best='mIoU')

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
