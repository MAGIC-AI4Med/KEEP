from tqdm import tqdm
import json
from utils import get_zeroshot_classifier, WSI_Classification_Dataset, zero_shot_prompt_select, cood2str, str2cood
import pandas as pd
from torch.utils.data import DataLoader
from load_keep import load_model
from segment_utils import zero_shot_segment
import numpy as np

label_dicts = {
    'camelyon_tumor_segment': {'Normal': 0,'Tumor': 1},
    'panda_all_segment': {'Normal': 0,'Tumor': 1},
    'aggc_tumor_segment': {'Normal': 0,'Tumor': 1},
}

prompt_file = '/Path/to/promt_file/'
csv_path = '/Path/to/test.csv'
embeddings_dir = '/Path/to/embedding/'
model_path = 'Path/to/model/'
mask_path = 'Path/to/gt_mask/'
save_dir = 'Path/to/save_dir/'
topn = 50
bootstrapping = 1000


df = pd.read_csv(csv_path) # Load split csv
device = 'cuda:0'
use_h5 = True
task = 'camelyon_tumor_segment'
workers = 8

with open(prompt_file, 'r') as pf: 
    prompts = json.load(pf)

## dataset
dataset = WSI_Classification_Dataset(
        df = df,
        data_source = embeddings_dir, 
        target_transform = None,
        index_col = 'slide_id',
        target_col = 'Diagnosis', 
        use_h5 = use_h5, 
        label_map = label_dicts[task]
    )
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers= workers, pin_memory=True)

## load model
KEEP_model = load_model(model_path)

## generate prompt classifier
merge_classifier = []
for prompt_idx in (pbar := tqdm(range(len(prompts)))):
    prompt = prompts[str(prompt_idx)]
    classifier = get_zeroshot_classifier(KEEP_model, test_dataloader, prompts, device)
    merge_classifier.append(classifier)

## select prompt classifier
ensemble_classifier = zero_shot_prompt_select(merge_classifier, test_dataloader, topn = topn, device = device)

## zero-shot evaluation
ensemble_results = zero_shot_segment(ensemble_classifier, test_dataloader, csv_path = csv_path, patch_size = 224, overlap = True, bootstrapping=bootstrapping, mask_path = mask_path, save_dir = save_dir, device= device)

dice_res = np.percentile(ensemble_results['mean_dice'], (2.5, 50, 97.5), interpolation='midpoint')
print('median mDice (2.5, 97.5) is %.3f (%.3f, %.3f). '%(dice_res[1],dice_res[0],dice_res[2]))
auc_res = np.percentile(ensemble_results['mean_auc'], (2.5, 50, 97.5), interpolation='midpoint')
print('median mAUC (2.5, 97.5) is %.3f (%.3f, %.3f). '%(auc_res[1],auc_res[0],auc_res[2]))