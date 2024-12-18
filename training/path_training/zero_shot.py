import logging
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from path_open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, retrieval_metrics, classification_metrics, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from .precision import get_autocast
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from path_open_clip.tokenizer import tokenize


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, cfg):
    autocast = get_autocast(cfg.MODEL.PRECISION)
    input_dtype = get_input_dtype(cfg.MODEL.PRECISION)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=cfg.DATALOADER.BATCH_SIZE):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def label2cap(cfg):
    with open(cfg.DATASET.ZEROSHOT_CLS_PROMPTS) as f:
        prompts = json.load(f)
    label_captions = dict()
    types = list(prompts['0']['classnames'].keys())
    for type_name in types:
        label_captions[type_name] = []
        for i in range(0,50):
            cls_name = prompts[str(i)]['classnames'][type_name]
            template = prompts[str(i)]['templates']
            cap = template.replace('CLASSNAME',cls_name)
            label_captions[type_name].append(cap)

    return label_captions

def contional_tokenize(text, tokenizer, cfg, args):
    if cfg.MODEL.KNOWLEDGE_GUIDANCE:
        text_input = dict()
        if cfg.MODEL.TEXT_ENCODER == 'bert':
            text_input['text_clip'] = tokenizer['bert'](list(text),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
        elif cfg.MODEL.TEXT_ENCODER in ['clip','biomed']:
            text_input['text_clip'] = tokenizer['clip'](text).to(device=args.device, non_blocking=True)
        text_input['text_knowledge'] = tokenizer['bert'](list(text),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
    elif cfg.MODEL.BERT_PRETRAIN is not None:
        text_input = tokenizer['bert'](list(text),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
    else:
        text_input = tokenizer['clip'](list(text)).to(device=args.device, non_blocking=True)
    
    return text_input
            

def zero_shot_eval(model, tokenizer, data, epoch, args, cfg):
    if 'zeroshot_cls' not in data and 'zeroshot_ret' not in data and 'zeroshot_po' not in data:
        return {}
    if cfg.SOLVER.ZEROSHOT_FREQUENCY == 0:
        return {}
    if (epoch % cfg.SOLVER.ZEROSHOT_FREQUENCY) != 0 and epoch != cfg.SOLVER.EPOCHS:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    if 'zeroshot_cls' in data:
        logging.info('Starting zero-shot classification...')
        autocast = get_autocast(cfg.MODEL.PRECISION)
        with torch.no_grad():
            model.eval()
            label_list = []
            image_embeddings = []
            for i, batch in enumerate(data['zeroshot_cls'].dataloader):
                images, labels = batch
                images = images.to(device=args.device)
                with autocast():
                    img_features = model.encode_image(images).detach().cpu().numpy()
                    image_embeddings.extend(img_features)
                    label_list.extend(labels)

            # unique_label = list(set(label_list))
            template_captions = label2cap(cfg)
            cap_embeddings = dict()
            for type_name,caps in template_captions.items():
                text_input = contional_tokenize(caps, tokenizer, cfg, args)
                cap_embeddings[type_name] = model.encode_text(text_input['text_clip'] if isinstance(text_input,dict) else text_input).detach().cpu().numpy()

        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        val_cls = []
        for i in range(50):
            cap_labels = []
            each_round = []
            for type_name,caps in cap_embeddings.items():
                text_embeddings = cap_embeddings[type_name][i,:]
                cap_labels.append(type_name)
                each_round.append(text_embeddings)
            each_round = np.array(each_round)
            each_round = each_round / np.linalg.norm(each_round, axis=1, keepdims=True)

            score = image_embeddings.dot(each_round.T)
            predictions = [cap_labels[np.argmax(i)] for i in score]

            cls_round = classification_metrics(label_list, predictions)
            val_cls.append(cls_round['WF1'])

        val_cls = np.array(val_cls)
        logging.info('Finish zero-shot classification.')

    if 'zeroshot_ret' in data:
        ## retrieval
        logging.info('Starting zero-shot retrieval...')
        autocast = get_autocast(cfg.MODEL.PRECISION)
        with torch.no_grad():
            model.eval()
            cap_list = []
            image_embeddings = []
            text_embeddings = []
            for i, batch in enumerate(data['zeroshot_ret'].dataloader):
                images, texts = batch
                images = images.to(device=args.device)
                if cfg.MODEL.KNOWLEDGE_GUIDANCE:
                    text_input = dict()
                    if cfg.MODEL.TEXT_ENCODER == 'bert':
                        text_input['text_clip'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                    elif cfg.MODEL.TEXT_ENCODER in ['clip','biomed']:
                        text_input['text_clip'] = tokenizer['clip'](texts).to(device=args.device, non_blocking=True)
                    text_input['text_knowledge'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                elif cfg.MODEL.BERT_PRETRAIN is not None:
                    text_input = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                else:
                    text_input = tokenizer['clip'](list(texts)).to(device=args.device, non_blocking=True)
                with autocast():
                    model_out = model(images, text_input)
                    image_features = model_out["image_features"].detach().cpu().numpy()
                    text_features = model_out["text_features"].detach().cpu().numpy()
                    image_embeddings.extend(image_features)
                    text_embeddings.extend(text_features)
      
        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        best_scores = []
        for tb in text_embeddings:
            arr = tb.dot(image_embeddings.T)
            best = arr.argsort()[-50:][::-1]
            best_scores.append(best)
        ##
        targets = list(range(0, len(image_embeddings)))
        val_ret = retrieval_metrics(targets, best_scores)
        logging.info('Finish zero-shot retrieval.')
    
    if 'zeroshot_po' in data:
        ## retrieval
        logging.info('Starting zero-shot pathout retrieval...')
        autocast = get_autocast(cfg.MODEL.PRECISION)
        with torch.no_grad():
            model.eval()
            cap_list = []
            image_embeddings = []
            text_embeddings = []
            for i, batch in enumerate(data['zeroshot_po'].dataloader):
                images, texts = batch
                images = images.to(device=args.device)
                if cfg.MODEL.KNOWLEDGE_GUIDANCE:
                    text_input = dict()
                    if cfg.MODEL.TEXT_ENCODER == 'bert':
                        text_input['text_clip'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                    elif cfg.MODEL.TEXT_ENCODER in ['clip','biomed']:
                        text_input['text_clip'] = tokenizer['clip'](texts).to(device=args.device, non_blocking=True)
                    text_input['text_knowledge'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                elif cfg.MODEL.BERT_PRETRAIN is not None:
                    text_input = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=args.device)
                else:
                    text_input = tokenizer['clip'](list(texts)).to(device=args.device, non_blocking=True)
                with autocast():
                    model_out = model(images, text_input)
                    image_features = model_out["image_features"].detach().cpu().numpy()
                    text_features = model_out["text_features"].detach().cpu().numpy()
                    image_embeddings.extend(image_features)
                    text_embeddings.extend(text_features)
      
        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        best_scores = []
        for tb in text_embeddings:
            arr = tb.dot(image_embeddings.T)
            best = arr.argsort()[-50:][::-1]
            best_scores.append(best)
        ##
        targets = list(range(0, len(image_embeddings)))
        val_po = retrieval_metrics(targets, best_scores)
        logging.info('Finish zero-shot pathout retrieval.')

    results = {}
    if 'zeroshot_cls' in data:
        res = np.percentile(val_cls, (25, 50, 75), interpolation='midpoint')
        # results['zeroshot-cls-WF1-mean'] = np.mean(val_cls)
        results['zeroshot-cls-WF1-median'] = res[1]
        # results['zeroshot-cls-WF1-std'] = np.std(val_cls)
        results['zeroshot-cls-WF1-Q1'] = res[0]
        results['zeroshot-cls-WF1-Q3'] = res[2]
    if 'zeroshot_ret' in data:
        results['zeroshot-ret-p@10'] = val_ret['p@10']
        results['zeroshot-ret-p@50'] = val_ret['p@50']
    if 'zeroshot_po' in data:
        results['zeroshot-po-p@10'] = val_po['p@10']
        results['zeroshot-po-p@50'] = val_po['p@50']

    return results
