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


def refine_seg(logits_all, coords_all, csv_path, patch_size = 224, overlap = True):
    
    df = pd.read_csv(csv_path)
    assert 'level0_mag' in df.columns, 'level0_mag column missing'
    
    df['slide_id'] = df['slide_id'].astype(str)
    
    preds_all_refined = dict()
    for k,v in tqdm(logits_all.items()):
        level0_mag = df['level0_mag'].values[df['slide_id']==k][0]
        
        preds_all_refined[k] = dict()
        
        coods_logits_dict = dict()
        cood_slide = coords_all[k]
        logits_slide = logits_all[k]
        
        coods_preds_dict = dict()
        for coods, logits in zip(cood_slide, logits_slide):
            if cood2str(coods) not in coods_logits_dict:
                coods_logits_dict[cood2str(coods)] = logits.cpu().numpy()
                max_v , pred_label = logits.max(0)
                coods_preds_dict[cood2str(coods)] = pred_label.item()
        overlap = level0_mag > 20
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
        
        preds_all_refined[k] = coods_preds_dict
        
    return preds_all_refined

def zero_shot_subtyping(classifier, dataloader, csv_path, bootstrapping = 1000, device = 'cuda', save_dir = './'):
    
    logits_all, coords_all, targets_all  = run(classifier, dataloader, device)
    
    print('Refine patch labels...')
    preds_all_refined = refine_seg(logits_all, coords_all, csv_path, patch_size = 256, overlap=False)
    
    print('Evaluate subtyping')
    gt_label = []
    pred_label = []
    all_preds = dict()
    for k,v in preds_all_refined.items():
        gt_label.append(targets_all[k])
        
        cls_fraction = []
        for ix in range(classifier.shape[1]):
            cls_fraction.append((np.array(list(v.values()))==ix).sum()/len(v))
            
        _, max_label = torch.tensor(cls_fraction[0:-1]).max(0)
        pred_label.append(max_label)
        all_preds[k] = max_label.item()

    json_str = json.dumps(targets_all, indent=2)
    with open(save_dir + 'gt_labels.json', 'w') as json_file:
        json_file.write(json_str)
    json_str = json.dumps(all_preds, indent=2)
    with open(save_dir + 'pred_probs.json', 'w') as json_file:
        json_file.write(json_str)

    boot_results = dict()
    boot_results['bacc'] = []
    boot_results['wF1'] = []
    len_sample = len(gt_label)
    for i in tqdm(range(bootstrapping)):
        random.seed(i)
        boot_gt = np.array(random.choices(list(gt_label), k = len_sample)) #[gt_labels[_id] for _id in boot_id]
        
        random.seed(i)
        boot_pre = np.array(random.choices(list(pred_label), k = len_sample))

        cm = confusion_matrix(boot_gt,boot_pre).tolist()
        bacc = balanced_accuracy_score(boot_gt, boot_pre)
        cls_rep = classification_report(boot_gt, boot_pre, output_dict=True, zero_division=0)
        
        boot_results['bacc'].append(bacc)
        boot_results['wF1'].append(cls_rep['weighted avg']['f1-score'])

    return boot_results