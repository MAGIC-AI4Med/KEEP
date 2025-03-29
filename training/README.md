# KEEP training Guide

## Prerequisites

Before training, you must prepare the following datasets and model weights:

### Data

#### Knowledge Data
- **Disease Ontology and UMLS**: These resources must be incorporated to construct a disease knowledge graph
- Example of the final KG can be found at: `./train_data/example_konwledge_graph.json`

#### Training Data
- **OpenPath and Quilt1M datasets**: Follow the data filtering process described in the KEEP paper to cluster image-text pairs into semantic groups
- Example of final semantic groups: `./train_data/example_pathology_vl_semantic_groups.csv`

#### Test Data
- **Downstream tile classification and retrieval datasets**
  - *Retrieval dataset example*: `./test_data/Arch_pubmed_test.csv`
  - *Classification dataset example*: `./test_data/Bach_test.csv`
  - *Classification prompt file*: `./test_data/Bach_prompt.json`

### Model Weights

| Model | Source |
|-------|--------|
| **PubmedBERT** | Available from [PubmedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| **KNOWLEDGE_BERT** | Available from [KnowledgeBERT](https://drive.google.com/file/d/1cVEOMDkn8x2375CrCbbgC0ih66fR1gC4/view?usp=sharing) |
| **UNI** | Available from [UNI](https://huggingface.co/MahmoodLab/UNI) |

---

> **Note**: Ensure all data follows the exact format shown in the example files to avoid training issues.

## Directory Structure
```
.
├── train_data/
│   ├── example_konwledge_graph.json
│   ├── example_pathology_vl_semantic_groups.json
│   └── image_folder
├── test_data/
│   ├── Arch_pubmed_test.csv
│   ├── Bach_test.csv
│   ├── Bach_prompt.json
│   └── image_folder
├── models/
│   ├── PubmedBERT/
│   ├── KNOWLEDGE_BERT/
└── └── UNI/