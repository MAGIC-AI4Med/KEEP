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

def zero_shot_segment(classifier, dataloader, csv_path, patch_size = 224, overlap = True, bootstrapping = 1000, mask_path = './', mask_ext = '_mask.tif', save_dir = './',  device = 'cuda'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    boot_results = dict()
    boot_results['mean_dice'] = []
    boot_results['mean_iou'] = []
    boot_results['mean_auc'] = []
    logits_all, coords_all = run(classifier, dataloader, device)
    
    print('Refine segmentation...')
    probs_all_refined = refine_seg(logits_all, coords_all, csv_path, patch_size = patch_size, overlap = overlap) 
    
    print('Evaluate segmentation via AUC')
    auc_all, best_thd = eval_seg_auc(probs_all_refined, mask_path, mask_ext = mask_ext, patch_size = patch_size, save_path = save_dir)
    
    print('Evaluate segmentation at best threshold from ROC...')
    dice_all, iou_all = eval_seg_coarse(probs_all_refined, mask_path, patch_size = patch_size, thd = best_thd)
    
    
    json_str = json.dumps(dice_all, indent=2)
    with open(save_dir + 'seg_dice_all.json', 'w') as json_file:
        json_file.write(json_str)
    json_str = json.dumps(iou_all, indent=2)
    with open(save_dir + 'seg_iou_all.json', 'w') as json_file:
        json_file.write(json_str)
    json_str = json.dumps(auc_all, indent=2)
    with open(save_dir + 'seg_auc_all.json', 'w') as json_file:
        json_file.write(json_str)
    
    for i in range(bootstrapping):
        random.seed(i)
        boot_dice = np.array(random.choices(list(dice_all.values()), k = len(dice_all)))
        boot_results['mean_dice'].append(boot_dice.mean())
        random.seed(i)
        boot_iou = np.array(random.choices(list(iou_all.values()), k = len(iou_all)))
        boot_results['mean_iou'].append(boot_iou.mean())
        random.seed(i)
        boot_auc = np.array(random.choices(list(auc_all.values()), k = len(auc_all)))
        boot_results['mean_auc'].append(boot_auc.mean())

    return boot_results


def refine_seg(logits_all, coords_all, csv_path, patch_size = 224, overlap = True):
    
    df = pd.read_csv(csv_path)
    assert 'level0_mag' in df.columns, 'level0_mag column missing'
    
    # preds_all_refined = dict()
    probs_all_refined = dict()
    for k,v in tqdm(logits_all.items()):
        level0_mag = df['level0_mag'].values[df['slide_id']==k][0]
        
        # preds_all_refined[k] = dict()
        probs_all_refined[k] = dict()
        
        coods_logits_dict = dict()
        cood_slide = coords_all[k]
        logits_slide = logits_all[k]
        
        coods_probs_dict = dict()
        for coods, logits in zip(cood_slide, logits_slide):
            if cood2str(coods) not in coods_logits_dict:
                coods_logits_dict[cood2str(coods)] = logits.cpu().numpy()
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
                coods_probs_dict[kk] = mean_logits[1].item()

        probs_all_refined[k] = coods_probs_dict
        
    return probs_all_refined

def eval_seg_auc(probs_all_refined, mask_path, mask_ext = '_mask.tif', patch_size = 224, save_path = './'):
        
    ## step 1. obtain patch label from GT masks
    if os.path.exists(save_path + 'mask_patch_gt.npy'):
        gt_patch_labels = np.load(save_path + 'mask_patch_gt.npy', allow_pickle=True).item()
    else:
        mask_list = os.listdir(mask_path)
        gt_patch_labels = dict()
        
        pbar = tqdm(total=len(probs_all_refined))
        for k,v in probs_all_refined.items():
            
            mask_name = os.path.join(k, mask_ext)
            
            if mask_name not in mask_list:  
                continue
            mask_wsi = openslide.open_slide(mask_path + mask_name) 
            gt_patch_labels[k] = dict()
            
            for kk,vv in v.items():
                coods = str2cood(kk)
                mask_img = np.array(mask_wsi.read_region(coods, 0, (patch_size, patch_size)).convert('L')) ## gray image for mask
                mask_nonzero = np.count_nonzero(mask_img)
                gt_patch_labels[k][kk] = 0
                if mask_nonzero > patch_size*patch_size/2:
                    gt_patch_labels[k][kk] = 1
            
            pbar.update(1)
        pbar.close()
            
        np.save(save_path + 'mask_patch_gt.npy', gt_patch_labels)
    
    ## step2. evaluation
    all_auc = dict()
    all_thd = []
    bad_k = []
    for k,v in gt_patch_labels.items():
        gt_per_slide = []
        probs_per_slide = []
        for kk,vv in v.items():
            coods = str2cood(kk)
            gt_per_slide.append(vv)
            # try:
            probs_per_slide.append(probs_all_refined[k][kk])
            # except:
                # bad_k.append(k)
        
        if len(np.unique(np.array(gt_per_slide))) ==1:
            continue
        
        auc_score = roc_auc_score(np.array(gt_per_slide), np.array(probs_per_slide))
        all_auc[k] = auc_score
        
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_per_slide), np.array(probs_per_slide))

        best_threshold = thresholds[np.argmax(tpr - fpr)]
        if not math.isinf(best_threshold):
            all_thd.append(best_threshold)
        else:
            continue        
        
    best_thd = np.array(all_thd).mean()
    return all_auc, best_thd


def eval_seg_coarse(probs_all_refined, mask_path, mask_ext = '_mask.tif', patch_size = 224, thd = 0.5):
    mask_list = os.listdir(mask_path)
    dice_all = dict()
    
    for k,v in tqdm(probs_all_refined.items()):
        mask_name = os.path.join(k, mask_ext)
            
        if mask_name not in mask_list:
            continue
        
        ## evaluate at level-16
        mask_wsi = openslide.open_slide(mask_path + mask_name)
        index = min(range(len(mask_wsi.level_downsamples)), key=lambda i: abs(mask_wsi.level_downsamples[i] - 16))
        mask_img = np.array(mask_wsi.read_region([0,0], index, mask_wsi.level_dimensions[index]).convert('L'))
        mag_num = int(mask_wsi.level_downsamples[index])

        mask_sum = np.count_nonzero(mask_img)*256
        pred_sum = 0
        intersection_sum = 0
        
        pred_mask = np.zeros_like(mask_img)
        for kk,vv in v.items():
            coods = str2cood(kk)

            if vv > thd:   
                pred_sum += patch_size*patch_size
                pred_mask[int(coods[1]/mag_num):int(coods[1]/mag_num+patch_size/mag_num), int(coods[0]/mag_num):int(coods[0]/mag_num+patch_size/mag_num)] = 255
        
        inter_mask = mask_img*pred_mask
        intersection_sum = np.count_nonzero(inter_mask)*256
        
        pred_sum = np.count_nonzero(pred_mask)*256
        
        safe_divide = mask_sum + pred_sum
        if safe_divide == 0:
            continue

        dice_per_slide = 2*intersection_sum/(mask_sum + pred_sum)
        dice_all[k] = dice_per_slide
        
    return dice_all
