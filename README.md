# KEEP

The official codes for **"A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis"**

[Preprint](https://arxiv.org/abs/2412.13126) | [Download Model](https://huggingface.co/Astaxanthin/KEEP) | [Webpage](https://loiesun.github.io/keep/) | [Cite](#reference)

---

**Abstract:** Deep learning has enabled the development of highly robust foundation models for various pathological tasks across diverse diseases and patient cohorts. Among these models, vision-language pre-training, which leverages large-scale paired data to align pathology image and text embedding spaces, and provides a novel zero-shot paradigm for downstream tasks. However, existing models have been primarily data-driven and lack the incorporation of domain-specific knowledge, which limits their performance in cancer diagnosis, especially for rare tumor subtypes. To address this limitation, we establish a **K**nowledg**E**-**E**nhanced **P**athology (**KEEP**) foundation model that harnesses disease knowledge to facilitate vision-language pre-training. Specifically, we first construct a disease knowledge graph (KG) that covers 11,454 human diseases with 139,143 disease attributes, including synonyms, definitions, and hypernym relations. We then systematically reorganize the millions of publicly available noisy pathology image-text pairs, into 143K well-structured semantic groups linked through the hierarchical relations of the disease KG. To derive more nuanced image and text representations, we propose a novel knowledge-enhanced vision-language pre-training approach that integrates disease knowledge into the alignment within hierarchical semantic groups instead of unstructured image-text pairs. Validated on 18 diverse benchmarks with more than 14,000 whole slide images (WSIs), KEEP achieves state-of-the-art performance in zero-shot cancer diagnostic tasks. Notably, for cancer detection, KEEP demonstrates an average sensitivity of 89.8% at a specificity of 95.0% across 7 cancer types, significantly outperforming vision-only foundation models and highlighting its promising potential for clinical application. For cancer subtyping, KEEP achieves a median balanced accuracy of 0.456 in subtyping 30 rare brain cancers, indicating strong generalizability for diagnosing rare tumors. All codes and models will be available for reproducing our results.

<img src="resources/teaser.png" alt="workflow" width="800" />


## News
**[12/18/2024]**: Code and model weights are now live. 

**[12/18/2024]**: We published the paper on ArXiv(https://arxiv.org/abs/2412.13126)


## What is KEEP? 
**KEEP** (**K**nowledg**E**-**E**nhanced **P**athology) is a foundation model designed for cancer diagnosis that integrates disease knowledge into vision-language pre-training. It utilizes a comprehensive disease knowledge graph (KG) containing 11,454 human diseases and 139,143 disease attributes, such as synonyms, definitions, and hierarchical relationships. KEEP reorganizes millions of publicly available noisy pathology image-text pairs into 143K well-structured semantic groups based on the hierarchical relations of the disease KG. By incorporating disease knowledge into the alignment process, KEEP achieves more nuanced image and text representations. The model is validated on 18 diverse benchmarks with over 14,000 whole-slide images (WSIs), demonstrating state-of-the-art performance in zero-shot cancer diagnosis, including an average sensitivity of 89.8% for cancer detection across 7 cancer types. KEEP also excels in subtyping rare cancers, achieving strong generalizability in diagnosing rare tumor subtypes.
- _**Why we need KNOWLEDGE?**_: We need knowledge integration in computational pathology to address the limitations of existing models, which often struggle with data scarcity, noisy annotations, and the complexity of cancer subtypes. Domain-specific knowledge, such as disease ontologies and medical terminologies, provides a structured framework that enhances the model’s understanding of pathology images by incorporating clinically relevant context. This knowledge helps improve model accuracy and generalizability, particularly in data-scarce or rare disease settings. Furthermore, it aids in distinguishing subtle disease manifestations, ensuring more precise diagnosis and better performance on downstream tasks like cancer detection and subtyping.


## Quick Start

You could directly download the models from google drive with link: [GoogleDrive](https://drive.google.com/drive/folders/1rzis8KJw4fdOyy2H3awYfAnAgByVXDLD?usp=sharing). These models also include functionality to extract patch embeddings for downstream tasks.

```python
cd ./quick_start
python keep_inference.py
```


## Evaluation on WSIs 
We provide a ```.py``` file for fast evaluation on WSIs as follows. You  need to change ```--data_path``` to the path where your WSIs are stored. In this part, you only need one 4090 GPU.

```bash
cd evaluation
python evaluation_wsis.py --data_path /path/to/wsi/
```


## Dataset Structuring
We manually annotate 1,000 noisy pathology images to fine-tune Yolov8. You can directly download the fine-tuned Yolov8 model from [Google Drive](https://drive.google.com/drive/my-drive) .

```bash
cd data

# detection pathology image in slide
python detection.py --data_path /path/to/images/ --model_path /path/to/yolov8/

# textual refinement: extract entities, paraphrased by templates
python text_processing.py --data_path /path/to/texts/ 

# image-text data cluster
python data_cluster.py --image_path /path/to/images/ --text_path /path/to/texts/ --structured_data_path /path/to/save/
``` 

## Knowledge Construction and Encoding
For knowledge graph construction, we download the knowledge structure from  [Disease Ontolog (DO)](https://disease-ontology.org/do/). Then, we search for synonyms in [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) based on the ```UMLS_CUI``` of each entity and construct the final **KG**.

For disease knowledge encoding, we train the knowldge encoder similar with our previous work [KEP](https://github.com/MAGIC-AI4Med/KEP). You could find more detailed information in the repository. In this part, we use four A100 GPUs.

## Vision-language Pre-training

### Installation
Start by cloning the repository and cd into the directory:

```bash
git clone https://github.com/MAGIC-AI4Med/KEEP.git
cd KEEP
```

Next, create a conda environment and install the dependencies:

```bash
conda create -n keep python=3.8 -y
conda activate keep
pip install --upgrade pip
pip install -r requirements.txt
```

### Training
If you need to retrain the model, you could refer to the following code and modify the relevant parameters. In this part, we only use one A100 GPU.

```bash
cd training

CUDA_VISIBLE_DEVICES=0
python main.py 
      --data-path /path/to/data/
      --save-path /path/to/save/
      --num-workers 8
      --batch-size 512
      --warmup 1000
```

## Performance Comparisons with Other Models

We present benchmark results for a range of representative tasks. A complete set of benchmarks can be found in the [paper](https://arxiv.org/abs/2412.18***). These results will be updated with each new iteration of KEEP. 

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
