from tqdm import tqdm
import json
from utils import get_zeroshot_classifier, zero_shot_prompt_select
from subtyping_utils import zero_shot_subtyping
from transformers import  AutoModel, AutoTokenizer
from torchvision import transforms
import h5py
import torch
import torch.nn.functional as F
import random


test_data_name = 'RCC'

prompt_file = './prompts/tcga_rcc_prompts.json'
h5_path = './h5_files/TCGA-RCC_examples/TCGA-BP-4161-01Z-00-DX1.a5c24186-a438-4c65-857e-d6da30340342.h5' # CCRCC
## TCGA-BP-4161-01Z-00-DX1.a5c24186-a438-4c65-857e-d6da30340342 CCRCC
## TCGA-KN-8425-01Z-00-DX1.1D2AB7D2-6AC3-4785-9FBC-40AEED5DE558 CHRCC
## TCGA-A4-7915-01Z-00-DX1.856b7fe5-bb58-48a4-a967-52d4e947a814 PRCC


model_path = 'Astaxanthin/KEEP' #'/path/to/keep/'
topn = 50
bootstrapping = 1000
 
device = 'cuda:0'
wsi_label = {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2}
id_label = {0:'CHRCC', 1:'CCRCC', 2:'PRCC'}
prompt_screening = True

with open(prompt_file, 'r') as pf: 
    prompts = json.load(pf)


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
    classifier = get_zeroshot_classifier(KEEP_model, wsi_label, prompt, device, add_normal=True)
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

subtyping_preds = zero_shot_subtyping(ensemble_classifier, tile_features, tile_coords, patch_size = 256,  overlap = True)

print('Predicted subtype: ' + id_label[subtyping_preds.item()])