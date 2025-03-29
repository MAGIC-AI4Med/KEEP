import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import jsonlines

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from tqdm import tqdm
import cv2
import json

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

templates = ['CLASSNAME.',
            'a photomicrograph showing CLASSNAME.',
            'a photomicrograph of CLASSNAME.',
            'an image of CLASSNAME.',
            'an image showing CLASSNAME.',
            'an example of CLASSNAME.',
            'CLASSNAME is shown.',
            'this is CLASSNAME.',
            'there is CLASSNAME.',
            'a histopathological image showing CLASSNAME.',
            'a histopathological image of CLASSNAME.',
            'a histopathological photograph of CLASSNAME.',
            'a histopathological photograph showing CLASSNAME.',
            'shows CLASSNAME.',
            'presence of CLASSNAME.',
            'CLASSNAME is present.',
            'an H&E stained image of CLASSNAME.',
            'an H&E stained image showing CLASSNAME.',
            'an H&E image showing CLASSNAME.',
            'an H&E image of CLASSNAME.',
            'CLASSNAME, H&E stain.',
            'CLASSNAME, H&E.'
            ]

sub_disease_nodes = {'DOID:0050117':'disease by infectious agent',
                  'DOID:7':'disease of anatomical entity',
                  'DOID:14566':'disease of cellular proliferation',
                  'DOID:150':'disease of mental health',
                  'DOID:0014667':'disease of metabolism',
                  'DOID:630':'genetic disease',
                  'DOID:0080015':'physical disorder',
                  'DOID:225':'syndrome'}

def get_random_hierarchy(all_do_nodes,sub_disease_nodes,node_id, use_syn = False):    
    if node_id == 'normal':
        return ['normal tissue', 'non-cancerous tissue', 'non-tumor tissue']
    
    cur_node_id = node_id
    if use_syn:
        cur_synname = [all_do_nodes[node_id]['name']] + all_do_nodes[node_id]['synonyms']
    else:
        cur_synname = [all_do_nodes[node_id]['name']]
    hierarchy_names = [cur_synname[random.randint(0,len(cur_synname)-1)]]
    
    if cur_node_id in sub_disease_nodes:
        return hierarchy_names

    while len(all_do_nodes[cur_node_id]['parent']) > 0:
        random_ix = random.randint(0,len(all_do_nodes[cur_node_id]['parent'])-1)
        par_node = all_do_nodes[cur_node_id]['parent'][random_ix]
        if use_syn:
            par_synname = [all_do_nodes[par_node]['name']] + all_do_nodes[par_node]['synonyms']
        else:
            par_synname =  [all_do_nodes[par_node]['name']]
        
        cur_node_id = par_node        
        if cur_node_id in sub_disease_nodes:
            break
        hierarchy_names.append(par_synname[random.randint(0,len(par_synname)-1)])
    
    return hierarchy_names

def get_hierarchy_cap(all_do_nodes,sub_disease_nodes,node_id, use_syn = False, mixed = False):
    hierarchy_names = get_random_hierarchy(all_do_nodes,sub_disease_nodes,node_id, use_syn)
    
    template = random.choices(templates, k = 1)[0]#'a histopathology image of '
    if isinstance(hierarchy_names, str):
        hierarchy_cap = hierarchy_names
    elif isinstance(hierarchy_names, list):
        reversed_names = hierarchy_names[::-1]
        hy_cap = template.replace('CLASSNAME', ' '.join(reversed_names))
        label_cap = template.replace('CLASSNAME', hierarchy_names[0])
        # hy_cap = ' '.join(reversed_names)
        # label_cap = hierarchy_names[0]
        if mixed:
            if random.random() > 0.5:
                hierarchy_cap = hy_cap  
            else:
                hierarchy_cap = label_cap
        else:
            hierarchy_cap = hy_cap
    return hierarchy_cap


class JsonDataset(Dataset):
    def __init__(self, input_group, knowledge_file, transforms, img_dir, num_instance, text_drop, preload_alldata = None, is_train = True, labeled_cap = 'both'):
        logging.debug(f'Loading json data from {input_group}.')
        with open(input_group) as f:
            self.group_cap_img_label = json.load(f)
        
        if knowledge_file:
            with open(knowledge_file) as f:
                self.knowledge_nodes = json.load(f)
        else:
            self.knowledge_nodes = None
        
        self.img_dir = img_dir
        groups = list(self.group_cap_img_label.keys())
        self.groups =groups
        
        ## select captions with label or without label
        if labeled_cap in ['label', 'unlabel']:
            labeled_groups = []
            unlabeled_groups = []
            for item in groups:
                if len(self.group_cap_img_label[item]['labels']):
                    labeled_groups.append(item)
                else:
                    unlabeled_groups.append(item)
            self.groups = []
            if labeled_cap == 'label': 
                self.groups = labeled_groups
            elif labeled_cap == 'unlabel':
                self.groups = unlabeled_groups
        
        self.is_train = is_train
        self.transforms = transforms
        self.num_instance = num_instance
        self.text_drop = text_drop
        
        self.random_captions=[]
        self.drop_captions = []
        self.repeated_groups = []
        for item in self.groups:
            self.repeated_groups.extend([item]*self.num_instance)
            try:
                caps = list(self.group_cap_img_label[item]['merged_caption'])
            except:
                caps = list(self.group_cap_img_label[item]['captions'])
            for i in range(self.num_instance):
                rand_id = random.randint(0,len(caps)-1)
                self.random_captions.append(caps[rand_id])
                drop_item = self.dropout(caps[rand_id])  
                self.drop_captions.append(drop_item)
        
        logging.debug('Done loading data.')

        self.preload_alldata = preload_alldata
        
    def __len__(self):
        return len(self.random_captions)
    
    def dropout(self,sentence, p=0.4):
        if np.random.rand() < 0.5:
            sentence = sentence.replace('  ',' ')
            return sentence
        
        words = sentence.split(' ')
        drop_len = round(len(words)*p)
        index = np.random.choice(len(words), drop_len)
        for i in index:
            words[i] = ''
        drop_st = ' '.join(words)
        drop_st = drop_st.replace('  ',' ')
        if drop_st.startswith(' '):
            drop_st = drop_st[1:]
        
        return drop_st
    
    def shuffle_data(self,):
        random.shuffle(self.groups)
        self.random_captions =[]
        self.drop_captions =[]
        self.repeated_groups = []
        for item in self.groups:
            self.repeated_groups.extend([item]*self.num_instance)
            try:
                caps = list(self.group_cap_img_label[item]['merged_caption'])
            except:
                caps = list(self.group_cap_img_label[item]['captions'])
            for i in range(self.num_instance):
                rand_id = random.randint(0,len(caps)-1)
                self.random_captions.append(caps[rand_id])
                drop_item = self.dropout(caps[rand_id])   ## to continue
                self.drop_captions.append(drop_item)

        logging.info(self.groups[0])
        

    def __getitem__(self, idx):
        
        if self.text_drop:
            text = self.drop_captions[idx]
        else:
            text = self.random_captions[idx]
            
        img_list = self.group_cap_img_label[self.repeated_groups[idx]]['images']
        if isinstance(img_list,dict):
            img_list = img_list['images']
        random.shuffle(img_list)
        img_name = img_list[0]
        
        if self.knowledge_nodes:
            label_list = list(self.group_cap_img_label[self.repeated_groups[idx]]['labels'].keys())
            random.shuffle(label_list)
            cap_label = label_list[0]
            if cap_label != 'unknown':
                hierarchy_cap = get_hierarchy_cap(self.knowledge_nodes, sub_disease_nodes, cap_label, use_syn=True,mixed=True)
                text = text if random.randint(0,1) == 1 else hierarchy_cap
        
        if self.preload_alldata is not None:
            if img_name in self.preload_alldata:
                img = self.preload_alldata[img_name]
            else:
                img = Image.open(os.path.join(str(self.img_dir), str(img_name)))
        else:
            img_root = str(self.img_dir)
            
            img_path = os.path.join(img_root, str(img_name))
            try:
                img = Image.open(os.path.join(img_root, str(img_name)))
            except:
                img = Image.open(os.path.join(img_root, str(img_name).split('-')[0],str(img_name)))
        
        image = self.transforms(img)
        
        if self.knowledge_nodes:
            return image, text, cap_label
        else:
            return image, text

def preload_dataset(cfg):
    img_file_path = cfg.DATASET.TRAIN_DATA
    if img_file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(img_file_path, sep='|')
        image_list = df[cfg.DATASET.CSV_IMG_KEY].tolist()
    elif img_file_path.split('.')[-1] == 'json':
        with open(img_file_path) as f:
            cap_imgs = json.load(f)
        image_list = []
        for k,v in cap_imgs.items():
            if isinstance(v,dict):
                image_list.extend(v['images'])
            else:
                image_list.extend(v)
    
    preload_img_data = {}
    logging.info('Preload the entire dataset...')

    # tot = ToTensor()
    
    for idx in tqdm(range(len(image_list))):
        img_dir = os.path.join(str(cfg.DATASET.IMG_DIR), str(image_list[idx]))
        
        if image_list[idx] not in preload_img_data:
            img = cv2.imread(img_dir)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            preload_img_data[image_list[idx]] = img

    print(r'The number of images is: %d '%(len(preload_img_data)))
    return preload_img_data


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_dir, img_key, caption_key, sep='|', text_drop = False, preload_alldata = None, is_train = True):
        logging.debug(f'Loading csv data from {input_filename}.')
        if sep == 'both':
            try:
                df = pd.read_csv(input_filename)
                test = df['image_name']
            except:
                df = pd.read_csv(input_filename, sep='\t', engine='python')
        else:
            df = pd.read_csv(input_filename, sep=sep, engine='python')

        self.img_dir = img_dir
        self.images = df[img_key].tolist()
        self.groups = df[caption_key].tolist()
        self.text_drop = text_drop
        self.is_train = is_train

        self.transforms = transforms
        logging.debug('Done loading data.')

        # self.tokenize = tokenizer
        self.preload_alldata = preload_alldata

    def __len__(self):
        return len(self.groups)
    
    def dropout(self,sentence, p=0.4):
        if np.random.rand() < 0.5:
            sentence = sentence.replace('  ',' ')
            return sentence
        
        words = sentence.split(' ')
        drop_len = round(len(words)*p)
        index = np.random.choice(len(words), drop_len)
        for i in index:
            words[i] = ''
        drop_st = ' '.join(words)
        drop_st = drop_st.replace('  ',' ')
        if drop_st.startswith(' '):
            drop_st = drop_st[1:]
        
        return drop_st

    def __getitem__(self, idx):
        if self.preload_alldata is not None:
            # image = np.asarray(bytearray(self.preload_alldata['imgs'][self.images[idx]]), dtype="uint8")
            # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # images = self.transforms(image)
            if self.images[idx] in self.preload_alldata:
                img = self.preload_alldata[self.images[idx]]
            else:
                img = Image.open(os.path.join(str(self.img_dir), str(self.images[idx])))

        else:
            # img_name = str(self.images[idx]).split('.')[0] + '.npy'
            img_name = os.path.join(str(self.img_dir), str(self.images[idx]))
            if not os.path.exists(img_name):
                test = 1
            # img = Image.open(img_name)
            try:
                img = Image.open(os.path.join(str(self.img_dir), str(self.images[idx])))
            except:
                img = Image.open(os.path.join(str(self.img_dir), str(self.images[idx]).split('-')[0],str(self.images[idx])))
            # img = Image.fromarray(img.astype('uint8'))
        
        images = self.transforms(img)
        
        if self.text_drop:
            texts = self.dropout(str(self.groups[idx]))
        else:
            texts = str(self.groups[idx])
        
        return images, texts

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataset: Dataset
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, cfg, preprocess_fn, is_train):
    input_filename = cfg.DATASET.TRAIN_DATA if is_train else cfg.DATASET.VAL_DATA
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_dir=cfg.DATASET.IMG_DIR,
        img_key=cfg.DATASET.CSV_IMG_KEY,
        caption_key=cfg.DATASET.CSV_CAPTION_KEY,
        sep=cfg.DATASET.CSV_SEPARATOR,
        text_drop = cfg.DATALOADER.TEXT_DROP,
        preload_alldata = args.preload_data if is_train else None,
        is_train = is_train,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataset, dataloader, sampler)

def get_json_dataset(args, cfg, preprocess_fn, is_train):
    input_filename = cfg.DATASET.TRAIN_DATA if is_train else cfg.DATASET.VAL_DATA
    assert input_filename
    dataset = JsonDataset(
        input_filename, 
        knowledge_file = cfg.DATASET.KNOWLEDGE_FILE,
        transforms=preprocess_fn,
        img_dir=cfg.DATASET.IMG_DIR,
        num_instance = cfg.DATALOADER.BATCH_SIZE//cfg.DATALOADER.CAPTION_NUM,
        text_drop = cfg.DATALOADER.TEXT_DROP,
        preload_alldata = args.preload_data if is_train else None,
        is_train = is_train,
        labeled_cap= cfg.DATASET.LABEL_CAP,
    )
    num_samples = len(dataset)
    # sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    # shuffle = is_train and sampler is None
    
    assert cfg.DATALOADER.BATCH_SIZE % cfg.DATALOADER.CAPTION_NUM == 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle= None,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        # sampler=sampler,
        drop_last=is_train,
        
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataset, dataloader)


def get_zeroshot_dataset(cfg, preprocess_fn, task):
    if task == 'classification':
        input_filename = cfg.DATASET.ZEROSHOT_CLS 
        imdir = cfg.DATASET.ZEROSHOT_CLS_IMDIR
        caption_key = 'label'
    elif task =='retrieval':
        input_filename = cfg.DATASET.ZEROSHOT_RET
        imdir = cfg.DATASET.ZEROSHOT_RET_IMDIR
        caption_key = 'caption'
    elif task =='po_retrieval':
        input_filename = cfg.DATASET.ZEROSHOT_PO
        imdir = cfg.DATASET.ZEROSHOT_PO_IMDIR
        caption_key = 'caption'
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_dir= imdir,
        img_key=cfg.DATASET.CSV_IMG_KEY,
        caption_key=caption_key,
        sep='both',
        preload_alldata = None,
        is_train=False
    )
    num_samples = len(dataset)
    sampler =  None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.WORKORS,
        pin_memory=True,
        sampler=sampler,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataset, dataloader, sampler)

def get_data(args, cfg, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if cfg.DATASET.TRAIN_DATA or cfg.DATASET.TYPE == "synthetic":
        data["train"] = get_dataset_fn(cfg.DATASET.TRAIN_DATA, cfg.DATASET.TYPE)(
            args, cfg, preprocess_train, is_train=True)

    if cfg.DATASET.VAL_DATA:
        data["val"] = get_dataset_fn(cfg.DATASET.VAL_DATA, cfg.DATASET.TYPE)(
            args, cfg, preprocess_val, is_train=False)

    if cfg.DATASET.ZEROSHOT_CLS:
        data['zeroshot_cls'] = get_zeroshot_dataset(cfg, preprocess_val, 'classification')

    if cfg.DATASET.ZEROSHOT_RET:
        data['zeroshot_ret'] = get_zeroshot_dataset(cfg, preprocess_val, 'retrieval')

    if cfg.DATASET.ZEROSHOT_PO:
        data['zeroshot_po'] = get_zeroshot_dataset(cfg, preprocess_val, 'po_retrieval')

    return data


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
