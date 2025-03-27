from tqdm import tqdm
import json
from utils import cood2str, str2cood
import pandas as pd
import torch
import torch.nn.functional as F
import os
import random
import numpy as np
import openslide
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math


@torch.no_grad()
def run(classifier, dataloader, device): 
    logits_all = dict()
    coords_all = dict()
    pbar = tqdm(total=len(dataloader))
    for idx, data in enumerate(dataloader): # batch size is always 1 WSI, 
        image_features = data['features'].to(device, non_blocking=True).squeeze(0)
        target = data['label'].to(device, non_blocking=True)
        coords = data['coords']
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        slide_id = dataloader.dataset.get_ids(idx)

        coords_all[slide_id] = coords
            
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ classifier

        correct_logits = torch.nn.functional.softmax(logits*10,1)

        logits_all[slide_id] = correct_logits

        pbar.update(1)
    
    pbar.close()

    return logits_all, coords_all

def zero_shot_segment(classifier, tile_features, tile_coords, mask_path, patch_size = 224, overlap = True):
    
    tile_features = F.normalize(tile_features, dim=-1)
    logits = tile_features @ classifier
    
    correct_logits = torch.nn.functional.softmax(logits*10,1)
    
    print('Refine segmentation...')
    probs_all_refined = refine_seg(correct_logits, tile_coords, patch_size = patch_size, overlap = overlap) 
    
    print('Evaluate segmentation via AUC')
    auc, best_thd = eval_seg_auc(probs_all_refined, mask_path, patch_size = patch_size)
    
    print('Evaluate segmentation at best threshold from ROC...')
    dice = eval_seg_coarse(probs_all_refined, mask_path, patch_size = patch_size, thd = best_thd)

    return auc, dice


def refine_seg(logits_slide, coords_slide, patch_size = 224, overlap = True):
    
    coods_logits_dict = dict()
    probs_all_refined = dict()
    for coods, logits in zip(coords_slide, logits_slide):
        if cood2str(coods) not in coods_logits_dict:
            coods_logits_dict[cood2str(coods)] = logits.cpu().numpy()
            probs_all_refined[cood2str(coods)] = logits[1].item()
            
    if overlap: 
        for kk,vv in coods_logits_dict.items():
            cur_logits = []
            lt, rt, lb, rb = [str2cood(kk)[0]-patch_size,str2cood(kk)[1]-patch_size], [str2cood(kk)[0], str2cood(kk)[1]-patch_size], [str2cood(kk)[0]-patch_size, str2cood(kk)[1]],str2cood(kk)
            if cood2str(lt) in coods_logits_dict:
                cur_logits.append(coods_logits_dict[cood2str(lt)])
            if cood2str(rt) in coods_logits_dict:
                cur_logits.append(coods_logits_dict[cood2str(rt)])
            if cood2str(lb) in coods_logits_dict:
                cur_logits.append(coods_logits_dict[cood2str(lb)])
            if cood2str(rb) in coods_logits_dict:
                cur_logits.append(coods_logits_dict[cood2str(rb)])
            
            cur_logits = np.array(cur_logits)
            mean_logits = cur_logits.mean(0)
            probs_all_refined[kk] = mean_logits[1].item()
        
    return probs_all_refined

def eval_seg_auc(probs_all_refined, mask_path, patch_size = 224, save_path = './'):
        
    ## step 1. obtain patch label from GT masks
    gt_patch_labels = dict()
    mask_wsi = openslide.open_slide(mask_path) 
    
    for kk,vv in probs_all_refined.items():
        coods = str2cood(kk)
        mask_img = np.array(mask_wsi.read_region(coods, 0, (patch_size, patch_size)).convert('L')) ## gray image for mask
        mask_nonzero = np.count_nonzero(mask_img)
        gt_patch_labels[kk] = 0
        if mask_nonzero > patch_size*patch_size/2:
            gt_patch_labels[kk] = 1
        
    ## step2. evaluation
    gt_per_slide = []
    probs_per_slide = []
    for kk,vv in gt_patch_labels.items():
        coods = str2cood(kk)
        gt_per_slide.append(vv)
        probs_per_slide.append(probs_all_refined[kk])
    
    auc_score = roc_auc_score(np.array(gt_per_slide), np.array(probs_per_slide))
    
    fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_per_slide), np.array(probs_per_slide))

    best_threshold = thresholds[np.argmax(tpr - fpr)]

    return auc_score, best_threshold


def eval_seg_coarse(probs_all_refined, mask_path,  patch_size = 224, thd = 0.5):

    ## evaluate at level-16
    mask_wsi = openslide.open_slide(mask_path)
    index = min(range(len(mask_wsi.level_downsamples)), key=lambda i: abs(mask_wsi.level_downsamples[i] - 16))
    mask_img = np.array(mask_wsi.read_region([0,0], index, mask_wsi.level_dimensions[index]).convert('L'))
    mag_num = int(mask_wsi.level_downsamples[index])

    mask_sum = np.count_nonzero(mask_img)*256
    pred_sum = 0
    intersection_sum = 0
    
    pred_mask = np.zeros_like(mask_img)
    for kk,vv in probs_all_refined.items():
        coods = str2cood(kk)

        if vv > thd:   
            pred_sum += patch_size*patch_size
            pred_mask[int(coods[1]/mag_num):int(coods[1]/mag_num+patch_size/mag_num), int(coods[0]/mag_num):int(coods[0]/mag_num+patch_size/mag_num)] = 255
    
    inter_mask = mask_img*pred_mask
    intersection_sum = np.count_nonzero(inter_mask)*256
    
    pred_sum = np.count_nonzero(pred_mask)*256
    
    safe_divide = mask_sum + pred_sum
    if safe_divide == 0:
        return 1

    dice_slide = 2*intersection_sum/(mask_sum + pred_sum)
        
    return dice_slide