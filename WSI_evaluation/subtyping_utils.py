from tqdm import tqdm
import json
from utils import cood2str, str2cood
import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report


@torch.no_grad()
def run(classifier, dataloader, device): 

    preds_all = dict()
    preds_all['all'] = []
    logits_all = dict()
    targets_all = dict()
    coords_all = dict()
    for idx, data in enumerate(dataloader):
        image_features = data['features'].to(device, non_blocking=True).squeeze(0)
        target = data['label']
        coords = data['coords']
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        slide_id = dataloader.dataset.get_ids(idx)

        coords_all[slide_id] = coords
            
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier
        logits_all[slide_id] = logits
        targets_all[slide_id] = target.item()

    return logits_all, coords_all, targets_all 


def refine_seg(logits_slide, coords_slide, patch_size = 224, overlap = True):
 
    coods_logits_dict = dict()
    coods_preds_dict = dict()
    for coods, logits in zip(coords_slide, logits_slide):
        if cood2str(coods) not in coods_logits_dict:
            coods_logits_dict[cood2str(coods)] = logits.cpu().numpy()
            max_v , pred_label = logits.max(0)
            coods_preds_dict[cood2str(coods)] = pred_label.item()
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
            max_v , pred_label = torch.from_numpy(mean_logits).max(0)
            coods_preds_dict[kk] = pred_label.item()
        
    return coods_preds_dict

def zero_shot_subtyping(classifier, tile_features, tile_coords, patch_size = 256,  overlap = True):
    
    tile_features = F.normalize(tile_features, dim=-1)
    logits = tile_features @ classifier
    
    correct_logits = torch.nn.functional.softmax(logits*10,1)
    
    print('Refine patch labels...')
    preds_all_refined = refine_seg(correct_logits, tile_coords, patch_size = patch_size, overlap=overlap)
    
    print('Evaluate subtyping')
    cls_fraction = []
    for ix in range(classifier.shape[1]):
        cls_fraction.append((np.array(list(preds_all_refined.values()))==ix).sum()/len(preds_all_refined))
            
    _, max_label = torch.tensor(cls_fraction[0:-1]).max(0)

    return max_label