from tqdm import tqdm
import json
from utils import get_zeroshot_classifier, WSI_Classification_Dataset, zero_shot_prompt_select
import pandas as pd
from torch.utils.data import DataLoader
from load_keep import load_model
from subtyping_utils import zero_shot_subtyping
import numpy as np

label_dicts = {
    'NSCLC_subtyping': {'LUAD': 0, 'LUSC': 1},
    'BRCA_subtyping': {'IDC': 0, 'ILC': 1},
    'RCC_subtyping': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2},
    'ESCA_subtyping': {'Adenocarcinoma, NOS': 0, 'Squamous cell carcinoma, NOS': 1},
    'Brain_subtyping': {'Glioblastoma': 0, 'Astrocytoma, NOS': 1, 'Oligodendroglioma, NOS': 2},
    'UBC_subtyping': {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4},
    'CPTAC_LUNG_subtyping': {'LUAD': 0, 'LUSC': 1},
    'ebrains_subtyping': {'Glioblastoma, IDH-wildtype': 0,
                      'Transitional meningioma':1,
                      'Anaplastic meningioma':2,
                      'Pituitary adenoma':3,
                      'Oligodendroglioma, IDH-mutant and 1p/19q codeleted':4,
                      'Haemangioma':5,
                      'Ganglioglioma':6,
                      'Schwannoma':7,
                      'Anaplastic oligodendroglioma, IDH-mutant, 1p/19q codeleted':8,
                      'Anaplastic astrocytoma, IDH-wildtype':9,
                      'Pilocytic astrocytoma':10,
                      'Angiomatous meningioma':11,
                      'Haemangioblastoma':12,
                      'Gliosarcoma':13,
                      'Adamantinomatous craniopharyngioma':14,
                      'Anaplastic astrocytoma, IDH-mutant':15,
                      'Ependymoma':16,
                      'Anaplastic ependymoma':17,
                      'Glioblastoma, IDH-mutant':18,
                      'Atypical meningioma':19,
                      'Metastatic tumours':20,
                      'Meningothelial meningioma':21,
                      'Langerhans cell histiocytosis':22,
                      'Diffuse large B-cell lymphoma of the CNS':23,
                      'Diffuse astrocytoma, IDH-mutant':24,
                      'Secretory meningioma':25,
                      'Haemangiopericytoma':26,
                      'Fibrous meningioma':27,
                      'Lipoma':28,
                      'Medulloblastoma, non-WNT/non-SHH':29},
}

prompt_file = '/Path/to/promt_file/'
csv_path = '/Path/to/test.csv'
embeddings_dir = '/Path/to/embedding/'
model_path = 'Path/to/model/'
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

ensemble_results = zero_shot_subtyping(ensemble_classifier, test_dataloader, csv_path = csv_path, bootstrapping=bootstrapping, device=device, save_dir = save_dir)

wf1_res = np.percentile(ensemble_results['wF1'], (2.5, 50, 97.5), interpolation='midpoint')
print('Zero-shot: median wF1 (2.5, 97.5) is %.3f (%.3f, %.3f). '%(wf1_res[1],wf1_res[0],wf1_res[2]))
bacc_res = np.percentile(ensemble_results['bacc'], (2.5, 50, 97.5), interpolation='midpoint')
print('Zero-shot: median bacc (2.5, 97.5) is %.3f (%.3f, %.3f). '%(bacc_res[1],bacc_res[0],bacc_res[2]))