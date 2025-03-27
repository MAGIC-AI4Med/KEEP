import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from path_open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from path_open_clip.tokenizer import tokenize
import random
from path_training.data_proc_group import get_hierarchy_cap
import copy

sub_disease_nodes = {'DOID:0050117':'disease by infectious agent',
                  'DOID:7':'disease of anatomical entity',
                  'DOID:14566':'disease of cellular proliferation',
                  'DOID:150':'disease of mental health',
                  'DOID:0014667':'disease of metabolism',
                  'DOID:630':'genetic disease',
                  'DOID:0080015':'physical disorder',
                  'DOID:225':'syndrome'}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def backward(total_loss, scaler):
    if scaler is not None:  
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
        
def generate_negative_text(knowledge_root, label_node, N_ins, N_neg):
    
    with open(knowledge_root) as f:
        all_do_nodes = json.load(f)
    # node_graph = dict()
    # for kk,vv in all_do_nodes.items():
    #     node_graph[kk] = vv['parent']
        
    unique_node = label_node[::N_ins]
    
    negative_text = []
    for node in unique_node:
        
        if node not in all_do_nodes:
            negative_text.extend(['unknown']*N_neg)
            continue
        
        if 'parent' not in all_do_nodes[node]:
            negative_text.extend(['unknown']*N_neg)
            continue
        
        parent_node = all_do_nodes[node]['parent']
        neg_children_nodes = []
        for par_node in parent_node:
            c_node = copy.deepcopy(all_do_nodes[par_node]['children'])
            if node not in c_node:
                print('node: ', node)
                print('par_node: ', c_node)
            c_node.remove(node)
            neg_children_nodes.extend(c_node)
            
        if len(neg_children_nodes) == 0:
            negative_text.extend(['unknown']*N_neg)
            continue

        random.shuffle(neg_children_nodes)
        if len(neg_children_nodes) > N_neg:
            select_children = neg_children_nodes[:N_neg]
        else:
            select_children = random.choices(neg_children_nodes, k=N_neg)
        
        for child in select_children:
            negative_text.append(get_hierarchy_cap(all_do_nodes,sub_disease_nodes,child, use_syn = True, mixed = True))
        
    return negative_text
    
        
def train_one_epoch(model, data, tokenizer, loss, epoch, optimizer, scaler, scheduler, args, cfg, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(cfg.MODEL.PRECISION)
    input_dtype = get_input_dtype(cfg.MODEL.PRECISION)


    model.train()
    # if args.distill:
    #     dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // cfg.SOLVER.ACCUM_FREQ
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if cfg.SOLVER.ACCUM_FREQ > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # logging.info(len(batch))
        i_accum = i // cfg.SOLVER.ACCUM_FREQ
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if cfg.DATASET.KNOWLEDGE_FILE:
            images, texts, cap_label = batch
            
            ## additional neg
            # N_ins = cfg.DATALOADER.BATCH_SIZE // cfg.DATALOADER.CAPTION_NUM
            # N_neg = N_ins
            # neg_texts = generate_negative_text(cfg.DATASET.KNOWLEDGE_FILE, cap_label, N_ins, N_neg)
            # texts = [item for item in texts] + neg_texts
        else:
            images, texts = batch

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        
        # text_labels = text2label(texts)

        if cfg.MODEL.KNOWLEDGE_GUIDANCE:
            text_input = dict()
            if cfg.MODEL.TEXT_ENCODER == 'bert':
                text_input['text_clip'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
            elif cfg.MODEL.TEXT_ENCODER in ['clip','biomed']:
                text_input['text_clip'] = tokenizer['clip'](texts).to(device=device, non_blocking=True)
            text_input['text_knowledge'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
        elif cfg.MODEL.BERT_PRETRAIN is not None:
            text_input = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
        elif cfg.MODEL.PRETRAINED_IMAGE == 'biomed':
            text_input = tokenizer['biomed'](list(texts), context_length=256).to(device)
        else:
            text_input = tokenizer['clip'](texts).to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        loss_weight = None
        if cfg.MODEL.KNOWLEDGE_GUIDANCE and cfg.LOSS.ADAPTIVE:
            loss_weight = float(epoch)/float(cfg.SOLVER.EPOCHS)*cfg.LOSS.WEIGHT[1]

        if cfg.SOLVER.ACCUM_FREQ == 1:
            with autocast():
                
                img_feat = model.encode_image(images)
                text_feat = model.encode_text(text_input)
                logit_scale = model.logit_scale.exp()
                if cfg.DATASET.KNOWLEDGE_FILE and cfg.MODEL.TYPE == 'hierarchy_metric':
                    losses = loss(img_feat, text_feat, cap_label, logit_scale, output_dict=True)
                else:
                    losses = loss(img_feat, text_feat, logit_scale, output_dict=True)
                    
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, text_input)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(text_input)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % cfg.SOLVER.ACCUM_FREQ) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(cfg.SOLVER.ACCUM_FREQ):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if cfg.MODEL.GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MODEL.GRAD_CLIP_NORM, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if cfg.MODEL.GRAD_CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MODEL.GRAD_CLIP_NORM, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.MODEL.GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MODEL.GRAD_CLIP_NORM, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if cfg.SOLVER.ACCUM_FREQ > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % cfg.SOLVER.LOG_EVERY_N_STEPS == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * cfg.SOLVER.ACCUM_FREQ * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = cfg.SOLVER.ACCUM_FREQ * cfg.DATALOADER.BATCH_SIZE * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = cfg.SOLVER.ACCUM_FREQ * cfg.DATALOADER.BATCH_SIZE / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, tokenizer, epoch, args, cfg, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, tokenizer, data, epoch, args, cfg)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(cfg.MODEL.PRECISION)
    input_dtype = get_input_dtype(cfg.MODEL.PRECISION)

    if 'val' in data and (cfg.SOLVER.VAL_FREQUENCY and (epoch % cfg.SOLVER.VAL_FREQUENCY) == 0):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features= [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                if cfg.MODEL.KNOWLEDGE_GUIDANCE:
                    text_input = dict()
                    if cfg.MODEL.TEXT_ENCODER == 'bert':
                        text_input['text_clip'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
                    elif cfg.MODEL.TEXT_ENCODER in ['clip','biomed']:
                        text_input['text_clip'] = tokenizer['clip'](texts).to(device=device, non_blocking=True)
                    text_input['text_knowledge'] = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
                elif cfg.MODEL.BERT_PRETRAIN is not None:
                    text_input = tokenizer['bert'](list(texts),add_special_tokens=True,max_length=256,pad_to_max_length=True,return_tensors='pt').to(device=device)
                else:
                    text_input = tokenizer['clip'](list(texts)).to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, text_input)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())

                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
