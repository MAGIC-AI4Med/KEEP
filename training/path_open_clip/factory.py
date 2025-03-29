import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model.model import KEP, vit_bert, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype
from .loss import  ClipLoss, KepMetricLoss, MetricLoss, HyMetricLoss
from .openai import load_openai_model
from .pretrained import download_pretrained_from_hf
from .transform import image_transform, AugmentationCfg
from .tokenizer import HFTokenizer, tokenize
from transformers import AutoTokenizer
import numpy as np
from torch import nn
import open_clip
from .model.ctran import ctranspath
from torchvision.models.resnet import Bottleneck
from .model.res_ssl import ResNetTrunk
from .model.pmc_clip import PMC_CLIP
import timm
from torchvision import transforms


HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name, bert_pretrain, text_encoder):
    tokenizer = dict()
    if bert_pretrain and text_encoder == 'bert':
        tokenizer['bert'] = AutoTokenizer.from_pretrained(bert_pretrain,do_lower_case=True, local_files_only=True)
        return tokenizer

    if model_name.startswith(HF_HUB_PREFIX):
        tokenizer['clip'] = HFTokenizer(model_name[len(HF_HUB_PREFIX):])
    else:
        config = get_model_config(model_name)
        tokenizer['clip'] = HFTokenizer(
            config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
        
    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def biomed_load_checkpoint(model, checkpoint_path, strict=True):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        from open_clip.big_vision import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    open_clip.model.resize_pos_embed(state_dict, model)
    open_clip.model.resize_text_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        logit_scale,
        text_encoder: Optional[str] = None,
        test_emb_dim: int = 512,
        bert_pretrain: Optional[str] = None,
        image_encoder: str = None,
        pretrained_image: str = None,
        knowledge_bert: Optional[str] = None,
        visual_embedding_head: bool = False,
        text_embedding_head: bool = False,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
):
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        pretrained_cfg = config['preprocess_cfg']
        model_cfg = config['model_cfg']
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        pretrained_cfg = {}
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')


    is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
    cast_dtype = get_cast_dtype(precision)

    model_cfg['embed_dim'] = test_emb_dim
    model = vit_bert(**model_cfg, 
                        text_encoder = text_encoder,
                        image_encoder = image_encoder,
                        bert_pretrain= bert_pretrain, 
                        cast_dtype=cast_dtype,
                        visual_embedding_head=visual_embedding_head,
                        text_embedding_head=text_embedding_head,
                        logit_scale= logit_scale
    )

    ## Whether to initialize the text encoder of KEP by the knowledge-BERT
    if knowledge_bert:
        checkpoint = torch.load(knowledge_bert, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        if next(iter(state_dict.items()))[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.text.load_state_dict(state_dict, strict=False)
        print('load knowledge to text encoder, missing keys: ', missing_keys)
        print('load knowledge to text encoder, unexpected keys: ', unexpected_keys)
        logging.info(f'Load pretrained text bert success from: {knowledge_bert}')

   
    uni_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    uni_path = pretrained_image + 'pytorch_model.bin'
    uni_model.load_state_dict(torch.load(uni_path, map_location="cpu"), strict=True)
    model.visual = uni_model
    model.visual.image_size = 224
    
    model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale))
    model.to(device)

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)
            from .transformer import LayerNormFp32
            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)
            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    # set image / mean metadata from pretrained_cfg if available, or use default
    if pretrained_image and image_encoder in ['ctp','uni','prov']:
        model.visual.image_mean = (0.485, 0.456, 0.406)
        model.visual.image_std = (0.229, 0.224, 0.225)
    elif pretrained_image and image_encoder == 'res_ssl':
        model.visual.image_mean = (0.70322989, 0.53606487, 0.66096631)
        model.visual.image_std = (0.21716536, 0.26081574, 0.20723464)
    else:
        try:
            model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
            model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD
        except:
            model.clip.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
            model.clip.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)
    

    print(model.logit_scale.exp())
    return model


def create_loss(args,cfg):

    if cfg.MODEL.TYPE == 'hierarchy_metric':
        return HyMetricLoss(loss_type=cfg.LOSS.SUBTYPE,caption_num=cfg.DATALOADER.CAPTION_NUM, knowledge_root = cfg.DATASET.KNOWLEDGE_FILE)
    else:
        return ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )


def create_model_and_transforms(
        model_name: str,
        logit_scale,
        text_encoder: Optional[str] = None,
        test_emb_dim: int = 512,
        bert_pretrain: Optional[str] = None,
        image_encoder: str = None,
        pretrained_image: str = None,
        knowledge_bert: Optional[str] = None,
        visual_embedding_head: bool = False,
        text_embedding_head: bool = False,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
):
    model = create_model(
        model_name,
        logit_scale,
        text_encoder,
        test_emb_dim,
        bert_pretrain,
        image_encoder,
        pretrained_image,
        knowledge_bert,
        visual_embedding_head,
        text_embedding_head,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
        output_dict=output_dict,
    )
    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        require_pretrained=True,
    )

    if not return_transform:
        return model

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess
