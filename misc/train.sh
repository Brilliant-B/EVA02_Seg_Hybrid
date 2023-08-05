SEG_CONFIG=configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py
PRETRAIN_CKPT=pretrained/eva02_L_pt_m38m_p14to16.pt

python -m torch.distributed.launch \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --deterministic --gpu-ids 4 5 6 7 \
    --options model.backbone.pretrained=${PRETRAIN_CKPT}