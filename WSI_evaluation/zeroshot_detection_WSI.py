from tqdm import tqdm
import json
from utils import get_zeroshot_classifier, zero_shot_prompt_select
from detection_utils import zero_shot_detection
from transformers import  AutoModel, AutoTokenizer
from torchvision import transforms
import h5py
import torch
import torch.nn.functional as F
import random


prompt_file = './prompts/cptac_cm_prompts.json'
h5_path = './h5_files/CPTAC-CM_examples/C3N-02373-22.h5'
## C3N-02373-22 tumor WSI
## C3L-00967-27 normal WSI

model_path = 'Astaxanthin/KEEP' #'/path/to/keep/'
topn = 50

device = 'cuda:0'
threshold= 0.5
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
    
detection_preds = zero_shot_detection(ensemble_classifier, tile_features, tile_coords, patch_size = 256, overlap = False) 

print('Tumor probability: %.4f'%(detection_preds))