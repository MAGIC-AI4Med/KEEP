from tqdm import tqdm
import json
from utils import get_zeroshot_classifier, zero_shot_prompt_select
from segment_utils import zero_shot_segment
from transformers import  AutoModel, AutoTokenizer
from torchvision import transforms
import h5py
import torch
import torch.nn.functional as F
import random


test_data_name = 'camelyon_tumor'

prompt_file = './prompts/other_camelyon_tumor_prompts.json'
h5_path = './h5_files/camelyon_examples/test_040.h5'
model_path = 'Astaxanthin/KEEP' #'/path/to/keep/'
mask_path = './h5_files/camelyon_examples/test_040_mask.tif' 
topn = 50

device = 'cuda:0'
wsi_label = {'Normal': 0,'Tumor': 1}
prompt_screening = True

with open(prompt_file, 'r') as pf: 
    prompts = json.load(pf)

with h5py.File(h5_path, 'r') as f:
    tile_features = torch.from_numpy(f['features'][:]).to(device)
    tile_coords = f['coords'][:]

## load model
KEEP_model = dict()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
KEEP_model['model'] = model
KEEP_model['tokenizer'] = tokenizer
KEEP_model['transform'] = transform

## generate prompt classifier
merge_classifier = []
for prompt_idx in (pbar := tqdm(range(len(prompts)))):
    prompt = prompts[str(prompt_idx)]
    classifier = get_zeroshot_classifier(KEEP_model, wsi_label, prompt, device)
    merge_classifier.append(classifier)

## select prompt classifier
if prompt_screening:
    print('Rank prompts...')
    ensemble_classifier = zero_shot_prompt_select(merge_classifier, tile_features, topn = topn, device = device)
else:
    ensemble_cls = torch.zeros_like(classifier)
    cter = 0
    while cter < topn:
        random.seed(cter)
        rand_id = random.randint(0,len(merge_classifier)-1)
        ensemble_cls += merge_classifier[rand_id]
        cter += 1
    ensemble_classifier = F.normalize(ensemble_cls, p=2, dim=0)

## zero-shot evaluation
auc, dice = zero_shot_segment(ensemble_classifier, tile_features, tile_coords, mask_path, patch_size = 224, overlap = True)

print('AUROC: %.4f, Dice: %.4f'%(auc, dice))