import numpy as np
import torch
import torch.nn.functional as F
import os
import h5py
from tqdm import tqdm
import random


class WSI_Classification_Dataset(Dataset):
    def __init__(self, 
                 df, 
                 data_source, 
                 target_transform = None,
                 index_col = 'slide_id',
                 target_col = 'Diagnosis', 
                 use_h5 = True,
                 label_map = None):
        """
        Args:
        """
        self.label_map = label_map
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.target_transform = target_transform
        self.data = df
        self.use_h5 = use_h5

    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return str(self.data.loc[ids, self.index_col])

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]

    def __getitem__(self, idx):
        
        slide_id = str(self.get_ids(idx))
        label = self.get_labels(idx)

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.use_h5:
            feat_path = os.path.join(self.data_source, 'h5_files', slide_id + '.h5')
                
            with h5py.File(feat_path, 'r') as f:
                features = torch.from_numpy(f['features'][:])
                coords = torch.from_numpy(f['coords'][:])
        else:
            feat_path = os.path.join(self.data_source, 'pt_files', slide_id + '.pt')
            features = torch.load(feat_path)
            coords = []
        
        return {'features': features, 'coords': coords, 'label': label}


def zero_shot_classifier(KEEP_model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if isinstance(templates, list):
                texts = [template.replace('CLASSNAME', classname) for template in templates]
            elif isinstance(templates, str):
                texts = [templates.replace('CLASSNAME', classname)]
                
            text_inputs = KEEP_model.tokenizer(texts,max_length=256,padding='max_length',truncation=True, return_tensors='pt').to(device)    
            class_embeddings = KEEP_model.model.encode_text(text_inputs)[0]
            
            if len(class_embeddings.shape) == 1:
                class_embeddings = class_embeddings.unsqueeze(0)
            
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_zeroshot_classifier(model, dataloader, prompts, device):
    
    classnames = prompts['classnames']
    templates = prompts['templates']

    idx_to_class = {v:k for k,v in dataloader.dataset.label_map.items()}
    
    n_classes = len(idx_to_class)
    
    classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]

    classifier = zero_shot_classifier(model, classnames_text, templates, device) # num_classes x feat_dim
    
    return classifier


def rank_cls_score(logits):
    
    abnormal_max_v, abnormal_max_index = logits.max(1)
    values, indices = torch.topk(logits, k=logits.shape[1], dim=1, largest=True)
    second_largest = values[:, 1]

    diff = abnormal_max_v - second_largest
    complement = torch.abs(abnormal_max_v + second_largest - 1)
    cls_score = (diff-complement).mean()
        
    return cls_score.item()

def zero_shot_prompt_select(classifiers, dataloader, topn, device):
    
    preds_all_info = dict()
    print('Step 1. Computing logits and rank score for each classifier...')
        
    for idx, data in tqdm(enumerate(dataloader)): # batch size is always 1 WSI,             
        # if idx >1:
        #     break
        image_features = data['features'].to(device, non_blocking=True).squeeze(0)
        target = data['label'].to(device, non_blocking=True)
        coords = data['coords']
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        slide_id = dataloader.dataset.get_ids(idx)
        
        preds_all_info[slide_id] = dict()
        preds_all_info[slide_id]['coords'] = coords
        preds_all_info[slide_id]['preds'] = dict()

        image_features = F.normalize(image_features, dim=-1) 
        
        for k,cls in enumerate(classifiers):
            logits = image_features @ cls
            preds_all_info[slide_id]['preds'][str(k)] = dict()
            preds_all_info[slide_id]['preds'][str(k)]['cls_score'] = rank_cls_score(logits)

    ## compute score for each text prompt classifier
    cls_diversity_score = []
    for k,cls in enumerate(classifiers):
        cls_diversity = []
        for i in range(cls.shape[1]):
            for j in range(i+1,cls.shape[1]):
                cls_diversity.append((cls[:,i] @ cls[:,j]).item())
                cls_score = np.array(cls_diversity).mean()
        cls_diversity_score.append(abs(cls_score))
    
    prompt_classifier_score = []
    for k,cls in enumerate(classifiers):
        each_classifer_score = []
        for slide_id, v in preds_all_info.items():
            each_classifer_score.append(preds_all_info[slide_id]['preds'][str(k)]['cls_score'])
        prompt_classifier_score.append(np.array(each_classifer_score).mean()) ## no diversity

    sorted_score, index = torch.sort(torch.tensor(prompt_classifier_score),descending=True)
    
    merge_cls = torch.zeros_like(classifiers[0])
    for cls_index in index[0:topn]:
        merge_cls += classifiers[cls_index]
    merge_cls = F.normalize(merge_cls, p=2, dim=0)
        
    return merge_cls

def cood2str(cood):
    return str(cood[0]) + '_' + str(cood[1])
def str2cood(str):
    return [int(item) for item in str.split('_')]

def accuracy(logits, target, topk=(1,)):
    pred = logits.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
