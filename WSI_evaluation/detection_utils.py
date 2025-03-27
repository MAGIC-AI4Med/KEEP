from tqdm import tqdm
import json
from utils import cood2str, str2cood
import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


@torch.no_grad()
def run(classifier, dataloader, device): 
    
    logits_all = dict()
    targets_all = dict()
    coords_all = dict()
    for idx, data in enumerate(dataloader): # batch size is always 1 WSI, 
        image_features = data['features'].to(device, non_blocking=True).squeeze(0)
        target = data['label']
        coords = data['coords']
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        slide_id = dataloader.dataset.get_ids(idx)
        coords_all[slide_id] = coords
            
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier

        correct_logits = torch.nn.functional.softmax(logits*10,1)

        logits_all[slide_id] = correct_logits
        
        targets_all[slide_id] = target.item()
    
    return logits_all, coords_all, targets_all


def refine_seg(logits_slide, coords_slide, patch_size = 224, threshold = 0.5, overlap = True):
    
    coods_logits_dict = dict()
    coods_preds_dict = dict()
    coods_probs_dict = dict()
    for coods, logits in zip(coords_slide, logits_slide):
        if cood2str(coods) not in coods_logits_dict:
            coods_logits_dict[cood2str(coods)] = logits.cpu().numpy()
            pred_label = 0  ## normal
            if logits[1] > threshold:
                pred_label = 1-pred_label  # tumor
            coods_preds_dict[cood2str(coods)] = pred_label
            coods_probs_dict[cood2str(coods)] = logits[1].item()
            
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
            pred_label = 0  # default normal
            if mean_logits[1] > threshold:
                pred_label = 1-pred_label  # tumor
            coods_preds_dict[kk] = pred_label
            coods_probs_dict[kk] = mean_logits[1].item()
        
    return coods_preds_dict, coods_probs_dict

def calculate_metric(gt, pred): 
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    sensitivity = TP / float(TP+FN)
    specificity = TN / float(TN+FP) 
    
    return sensitivity, specificity

def zero_shot_detection(classifier, tile_features, tile_coords, patch_size = 256, overlap = False):

    tile_features = F.normalize(tile_features, dim=-1)
    logits = tile_features @ classifier
    
    correct_logits = torch.nn.functional.softmax(logits*10,1)
    
    print('Refine patch labels...')
    preds_refine, probs_refine = refine_seg(correct_logits, tile_coords, patch_size = patch_size, overlap = overlap) 
    
    # compute tumor ratio
    wsi_tumor_prob = np.array(list(preds_refine.values())).sum()/len(preds_refine)
    
    return wsi_tumor_prob