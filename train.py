import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, "
import os.path as osp
import time
import re
import loralib as lora

import torch
import torch.nn as nn
import mmcv
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from mmseg.models.backbones import EVA2



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args



'''
def find_eva_linear_names(model):
    lora_module_names = set()
    for name, module in model.backbone.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, model, checkpoint_dir=None):
    from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    )
    from peft import (
        prepare_model_for_kbit_training,
        LoraConfig,
        get_peft_model,
        PeftModel
    )
    from peft.tuners.lora import LoraLayer
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    import bitsandbytes as bnb
    
    pconfig=PretrainedConfig(is_encoder_decoder=True,torch_dtype=torch.float32)
    prtr=PreTrainedModel(pconfig)
    # prtr.save_pretrained('workbench/pretrained/')
    
    model = AutoModelForCausalLM.from_pretrained(
        None,
        state_dict=model.state_dict(),
        config=prtr,
        load_in_4bit=True,
        device_map='auto',
        max_memory={0: '5120MB'},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ),
        torch_dtype=torch.bfloat16,
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=torch.bfloat16
    model=prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, osp.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        print(f'Adding LoRA modules...')
        modules = find_eva_linear_names(model)
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=0.1,
            bias="none",
        )
        model=get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight') and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)

    return model
'''



def get_finetune_model(model, code, verbose=False):
    def freeze_match(name, f_list):
        ret = False
        for n in f_list:
            ret = ret or (re.search(n, name) is not None)
        return ret
    
    freeze_list = []
    
    if code == 1:
        checkpoint = torch.load(model.backbone.pretrained, map_location='cpu')["model"]
        freeze_list.extend([f"backbone.{key}" for key in checkpoint.keys()])
    
    if verbose:
        print("**frozen parameters**")
        print(f"List: {freeze_list}")
    for key, value in model.named_parameters():
        value.requires_grad = not freeze_match(key, freeze_list)
        if verbose:
            print(key, value.requires_grad)
    return model



def main(args, info, verbose=False):
    finetune_code, neck_name = info["finetune_code"], info["config_neck"]
    config = f"configs/{neck_name}_neck.py"
    cfg = Config.fromfile(config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
    cfg.work_dir = osp.join(cfg.work_dir, f"neck_{neck_name}_finetune_{finetune_code}")
    
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir.replace("/hy-tmp/", "")))
    # dump config
    cfg.dump(osp.join(cfg.work_dir.replace("/hy-tmp/", ""), osp.basename(config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir.replace("/hy-tmp/", ""), f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f"Neck: {neck_name}; Finetune_code: {finetune_code}")

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    if verbose:
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
        logger.info(f'Config:\n{cfg.pretty_text}')
    meta['env_info'] = env_info

    # set random seeds
    if args.seed is not None:
        if verbose:
            logger.info(f'Distributed training: {distributed}')
            logger.info(f'Set random seed to {args.seed}, deterministic: '
                        f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    # finetune_code: {0: non_freeze; 1: freeze_loaded_eva2}
    if finetune_code > 0:
        model = get_finetune_model(model, finetune_code, verbose=verbose)
    if verbose:
        logger.info(model)
    
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    args = parse_args()
    neck_choices = ["fpn", "sfp", "linear"]
    # finetune_code = {0: no-freeze, 1: freeze EVA}
    for n in neck_choices:
        hyper_info = {
            "finetune_code": 0,
            "config_neck": n,
        }
        main(args, hyper_info)