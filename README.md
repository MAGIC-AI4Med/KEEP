# KEEP (**K**nowledg**E**-**E**nhanced **P**athology)

The official codes for **"A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis"**

[Preprint](https://arxiv.org/abs/2412.13126) | [Hugging Face](https://huggingface.co/Astaxanthin/KEEP) | [Website](https://loiesun.github.io/keep/) | [Cite](#reference)

---

**Abstract:** Deep learning has enabled the development of highly robust foundation models for various pathological tasks across diverse diseases and patient cohorts. Among these models, vision-language pre-training, which leverages large-scale paired data to align pathology image and text embedding spaces, and provides a novel zero-shot paradigm for downstream tasks. However, existing models have been primarily data-driven and lack the incorporation of domain-specific knowledge, which limits their performance in cancer diagnosis, especially for rare tumor subtypes. To address this limitation, we establish a **K**nowledg**E**-**E**nhanced **P**athology (**KEEP**) foundation model that harnesses disease knowledge to facilitate vision-language pre-training. Specifically, we first construct a disease knowledge graph (KG) that covers 11,454 human diseases with 139,143 disease attributes, including synonyms, definitions, and hypernym relations. We then systematically reorganize the millions of publicly available noisy pathology image-text pairs, into 143K well-structured semantic groups linked through the hierarchical relations of the disease KG. To derive more nuanced image and text representations, we propose a novel knowledge-enhanced vision-language pre-training approach that integrates disease knowledge into the alignment within hierarchical semantic groups instead of unstructured image-text pairs. Validated on 18 diverse benchmarks with more than 14,000 whole slide images (WSIs), KEEP achieves state-of-the-art performance in zero-shot cancer diagnostic tasks. Notably, for cancer detection, KEEP demonstrates an average sensitivity of 89.8% at a specificity of 95.0% across 7 cancer types, significantly outperforming vision-only foundation models and highlighting its promising potential for clinical application. For cancer subtyping, KEEP achieves a median balanced accuracy of 0.456 in subtyping 30 rare brain cancers, indicating strong generalizability for diagnosing rare tumors. All codes and models will be available to reproduce our results.

<img src="resources/teaser.png" alt="workflow" width="800" />


## News
**[03/31/2025]**: Datasets, including manual annotation of pathology images, disease knowledge graph, and pathology image-text semantic groups for KEEP training are now released. 

**[03/28/2025]**: Codes for KEEP training are now updated. 

**[03/27/2025]**: Codes for single WSI inference on different tasks are now available. 

**[12/24/2024]**: Model weights for easy inference are now available on [Huggingface](https://huggingface.co/Astaxanthin/KEEP). 

**[12/23/2024]**: Model weights for pathology image detection are now available. 

**[12/23/2024]**: Codes for WSI evaluation are now available. 

**[12/18/2024]**: Model weights for easy inference are now available. 

**[12/18/2024]**: Paper is available in ArXiv (https://arxiv.org/abs/2412.13126).


## What is KEEP? 
**KEEP** (**K**nowledg**E**-**E**nhanced **P**athology) is a foundation model designed for cancer diagnosis that integrates disease knowledge into vision-language pre-training. It utilizes a comprehensive disease knowledge graph (KG) containing 11,454 human diseases and 139,143 disease attributes, such as synonyms, definitions, and hierarchical relationships. KEEP reorganizes millions of publicly available noisy pathology image-text pairs into 143K well-structured semantic groups based on the hierarchical relations of the disease KG. By incorporating disease knowledge into the alignment process, KEEP achieves more nuanced image and text representations. The model is validated on 18 diverse benchmarks with over 14,000 whole-slide images (WSIs), demonstrating state-of-the-art performance in zero-shot cancer diagnosis, including an average sensitivity of 89.8% for cancer detection across 7 cancer types. KEEP also excels in subtyping rare cancers, achieving strong generalizability in diagnosing rare tumor subtypes.
- _**Why we need KNOWLEDGE?**_ We need knowledge integration in computational pathology to address the limitations of existing models, which often struggle with data scarcity, noisy annotations, and the complexity of cancer subtypes. Domain-specific knowledge, such as disease ontologies and medical terminologies, provides a structured framework that enhances the model’s understanding of pathology images by incorporating clinically relevant context. This knowledge helps improve model accuracy and generalizability, particularly in data-scarce or rare disease settings. Furthermore, it aids in distinguishing subtle disease manifestations, ensuring more precise diagnosis and better performance on downstream tasks like cancer detection and subtyping.


## Quick Start
You can directly load **KEEP** from Huggingface and conduct inference with the following codes:
```python
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from PIL import Image

model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

example_image_path = './quick_start/example.tif'
example_text = ['an H&E image of breast invasive carcinoma.', 'an H&E image of normal tissue.', 'an H&E image of lung adenocarcinoma.']

img_input =  transform(Image.open(example_image_path).convert('RGB')).unsqueeze(0)
token_input = tokenizer(example_text,max_length=256,padding='max_length',truncation=True, return_tensors='pt')

img_feature = model.encode_image(img_input)
text_feature = model.encode_text(token_input)

```

Alternatively, you can download the model weights from Google Drive with the link: [model_weights](https://drive.google.com/drive/folders/1warXpxtb4PoL_fQPyT1qsbSAdWjCRy04?usp=sharing) for easy inference.


## Evaluation on WSI-level Tasks 
We provide a ```.py``` file for fast evaluation on WSIs, including cancer region segmentation, cancer detection, and cancer subtyping. Before evaluation, you need to follow [CLAM](https://github.com/mahmoodlab/CLAM) to extract patch-level features for the tested WSI and save them as h5 files. For instance, you can download exemplary h5 files from [KEEP_release](https://drive.google.com/drive/folders/1rzis8KJw4fdOyy2H3awYfAnAgByVXDLD?usp=sharing) and put the folder "h5_files" in "WSI_evaluation". In this part, you only need one 4090 GPU.

```bash
cd WSI_evaluation
python zeroshot_segmentation_WSI.py
python zeroshot_detection_WSI.py
python zeroshot_subtyping_WSI.py
```


## Pathology Image Detection
We manually annotate 1,000 noisy pathology images to fine-tune Yolov8. The annotations can be found in [KEEP_dataset](https://huggingface.co/datasets/Loie/KEEP_dataset). Alternatively, you can download the fine-tuned Yolov8 model weights directly from [PathDetector](https://drive.google.com/file/d/1CtQdGTrmMokUYaMczW1BsEr2BT8YkHZ2/view?usp=sharing).

```python
from ultralytics import YOLO

model = YOLO("yolov8x.yaml")  # build a new model from scratch
model = YOLO("./best.pt")  # load a pretrained model
img_path = 'example.jpg'
results = model(img_path)  # detect pathology regions on an image
```


## Knowledge Construction and Encoding
For knowledge graph construction, we download the knowledge structure from  [Disease Ontology (DO)](https://disease-ontology.org/do/). Then, we search for synonyms in [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) based on the ```UMLS_CUI``` of each entity and construct the final **KG**, which can also be downloaded from [KEEP_dataset](https://huggingface.co/datasets/Loie/KEEP_dataset) upon requests.

For disease knowledge encoding, we train the knowledge encoder in a way similar to our previous work [KEP](https://github.com/MAGIC-AI4Med/KEP). You can find more detailed information in the repository. In this part, we use 4 A100 GPUs.

## Vision-language Pre-training

### Installation

create a conda environment and install the dependencies:

```bash
cd training
conda create -n keep python=3.8 -y
conda activate keep
pip install -r requirements.txt
```

### Training
Before training, you need to collect pathology image-text pairs from [OpenPath](https://drive.google.com/drive/folders/1b5UT8BzUphkHZavRG-fmiyY9JWYIWZER) and [Quilt1M](https://zenodo.org/record/8239942), with seqcequent filtering and clustering pathology image-text pairs into semantic groups. The semantic groups can also be acquired from [KEEP_dataset](https://huggingface.co/datasets/Loie/KEEP_dataset). During training, you could refer to the following code and modify the relevant parameters in `training/configs/keep_config.yml`. In this part, we only use one A100 GPU.

```bash
cd training
python -m path_training.main
```

## Performance Comparisons with Other Models

We present benchmark results for a range of representative tasks. A complete set of benchmarks can be found in the [paper](https://arxiv.org/abs/2412.13126). These results will be updated with each new iteration of KEEP. 

### Zero-shot Cancer Region Segmentation (DICE) 
| Models | PLIP[[1]](https://www.nature.com/articles/s41591-023-02504-3) | QuiltNet [[2]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html) |   MI-Zero (Pub) [[3]](https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html) | CONCH [[4]](https://www.nature.com/articles/s41591-024-02856-4) | **KEEP(Ours)**  |
|:---------------|--------------:|---------------------:|-------------------------:|-----------------:|------------------:|
| CAMELYON16 | 0.253 | 0.157 | 0.186 | 0.292 | **0.361** |
| PANDA | 0.295 | 0.309 | 0.276 | 0.315 | **0.334** |
| AGGC22 | 0.284 | 0.282 | 0.324 | 0.449 | **0.530** |

### Zero-shot Cancer Detection (AUROC)
| Models | CHIEF[[1]](https://www.nature.com/articles/s41586-024-07894-z) |   PLIP [[2]](https://www.nature.com/articles/s41591-023-02504-3)   |   QuiltNet [[3]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html)     |   MI-Zero (Pub) [[4]](https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html) |   CONCH [[5]](https://www.nature.com/articles/s41591-024-02856-4) | KEEP(Ours) |
|:---------------|--------------:|--------------------:|-----------------:|-----------------:|------------------:| -----------------:|
| CPTAC-CM | 0.915 | 0.970 | 0.972 | 0.985 | **0.994** | **0.994** | 
| CPTAC-CCRCC | 0.723 | 0.330 | 0.755 | 0.886 | 0.871 | **0.999** |
| CPTAC-PDA | 0.825 | 0.391 | 0.464 | 0.796 | 0.920 | **0.929** |
| CPTAC-UCEC | 0.955 | 0.945 | 0.973 | 0.979 | 0.996 | **0.998** | 
| CPTAC-LSCC | 0.901 | 0.965 | 0.966 | 0.910 | **0.987** | 0.983 | 
| CPTAC-HNSCC | 0.946 | 0.898 | 0.874 | 0.918 | **0.982** | 0.976 | 
| CPTAC-LUAD | 0.891 | 0.988 | 0.991 | 0.981 | 0.999 | **1.000** |  

### Zero-shot Cancer Subtyping (BACC) 
| Models     | PLIP [[1]](https://www.nature.com/articles/s41591-023-02504-3) |   QuiltNet [[2]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html)      |   MI-Zero (Pub) [[3]](https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html)     |   CONCH [[4]](https://www.nature.com/articles/s41591-024-02856-4) |   **KEEP(Ours)**  |
|:---------------|--------------:|---------------------------:|-------------------------:|-----------------:|------------------:|
| TCGA-BRCA | 0.519 | 0.500 | 0.633 | 0.727 | **0.774** |
| TCGA-NSCLC | 0.699 | 0.667 | 0.753 | 0.901 | **0.902** |
| TCGA-RCC | 0.735 | 0.755 | 0.908 | 0.921 | **0.926** |
| TCGA-ESCA | 0.614 | 0.746 | 0.954 | 0.923 | **0.977** |
| TCGA-BRAIN | 0.361 | 0.346 | 0.361 | 0.453 | **0.604** |
| UBC-OCEAN | 0.343 | 0.469 | 0.652 | **0.674** | 0.661 |
| CPTAC-NSCLC | 0.647 | 0.607 | 0.643 | 0.836 | **0.863** |
| EBRAINS | 0.096 | 0.093 | 0.325 | 0.371 | **0.456** |


## Acknowledgements
The project was built on top of amazing repositories such as [MI-Zero](https://github.com/mahmoodlab/MI-Zero), [CLAM](https://github.com/mahmoodlab/CLAM), [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors and developers for their contribution.

## Reference
If you find our work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2412.13126):


```
@article{zhou2024keep,
  title={A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis},
  author={Xiao Zhou, Luoyi Sun, Dexuan He, Wenbin Guan, Ruifen Wang, Lifeng Wang, Xin Sun, Kun Sun, Ya Zhang, Yanfeng Wang, Weidi Xie},
  journal={arXiv preprint arXiv:2412.13126},
  year={2024}
}
``` 

<img src=resources/logo.png> 
