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


def refine_seg(logits_all, coords_all, csv_path, patch_size = 224, threshold = 0.5, overlap = True):
    
    df = pd.read_csv(csv_path)
    assert 'level0_mag' in df.columns, 'level0_mag column missing'
    
    preds_all_refined = dict()
    probs_all_refined = dict()
    for k,v in tqdm(logits_all.items()):
        level0_mag = df['level0_mag'].values[df['slide_id']==k][0]
        
        preds_all_refined[k] = dict()
        probs_all_refined[k] = dict()
        
        coods_logits_dict = dict()
        cood_slide = coords_all[k]
        logits_slide = logits_all[k]
        
        coods_preds_dict = dict()
        coods_probs_dict = dict()
        for coods, logits in zip(cood_slide, logits_slide):
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
        
        preds_all_refined[k] = coods_preds_dict
        probs_all_refined[k] = coods_probs_dict
        
    return preds_all_refined, probs_all_refined

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

def zero_shot_detection(classifier, dataloader, csv_path, bootstrapping = 1000, threshold = 0.0, device = 'cuda', save_dir = './'):

    if not isinstance(classifier, list):
        boot_results = dict()

        boot_results['auc'] = []
        boot_results['sensitivity'] = []
        boot_results['specificity'] = []
        boot_results['sensitivity_95'] = []
        boot_results['specificity_95'] = []
        logits_all, coords_all, targets_all = run(classifier, dataloader, device)
        
        print('Refine patch labels...')
        preds_all_refined, probs_all_refined = refine_seg(logits_all, coords_all, csv_path, patch_size = 256, threshold=threshold, overlap=False)  ## 112 for panda
        
        print('Evaluate detection via AUC')
        gt_label = []
        pred_prob = []
        all_preds = dict()
        for k,v in preds_all_refined.items():
            tumor_frac = np.array(list(v.values())).sum()/len(v)
            normal_frac = 1 - tumor_frac
            gt_label.append(targets_all[k])
            pred_prob.append(tumor_frac)
            all_preds[k] = tumor_frac
            
        # save_prediction(probs_all_refined, wsi_path, targets_all, all_preds, patch_size = 256, save_dir = save_dir)
            
        json_str = json.dumps(targets_all, indent=2)
        with open(save_dir + 'gt_labels.json', 'w') as json_file:
            json_file.write(json_str)
        json_str = json.dumps(all_preds, indent=2)
        with open(save_dir + 'pred_probs.json', 'w') as json_file:
            json_file.write(json_str)
        
        auc_all = roc_auc_score(np.array(gt_label), np.array(pred_prob))
        print('AUC across all slides are: %.4f'%(auc_all))
        
        for i in range(bootstrapping):
            
            random.seed(i)
            boot_gt = np.array(random.choices(gt_label, k = len(gt_label)))
            random.seed(i)
            boot_prob = np.array(random.choices(pred_prob, k = len(pred_prob)))
            boot_auc = roc_auc_score(boot_gt, boot_prob)
            boot_results['auc'].append(boot_auc)
    
    return boot_results