from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# Image address for training
_C.DATASET.IMG_DIR = 'Path/to/training/image/folder'
# Which type of dataset to process  ["csv", "synthetic", "auto"]
_C.DATASET.TYPE = 'csv'
# Training information
_C.DATASET.TRAIN_DATA = 'Path/to/training/data (.csv)'
# Validation information
_C.DATASET.VAL_DATA = 'Path/to/validation/data (.csv)'
# Image address for Zeroshot evaluation 
_C.DATASET.ZEROSHOT_CLS_IMDIR = 'Path/to/test (zeroshot classification)/image/folder'
# Zeroshot information 
_C.DATASET.ZEROSHOT_CLS = 'Path/to/test (zeroshot classifation)/data (.csv)'

_C.DATASET.ZEROSHOT_CLS_PROMPTS = 'Path/to/test (zeroshot classifation)/prompts (.json)'

# Image address for retrieval
_C.DATASET.ZEROSHOT_RET_IMDIR = 'Path/to/test (retrieval)/image/folder'
# Retrieval information 
_C.DATASET.ZEROSHOT_RET = 'Path/to/test (retrieval)/data (.csv)'
# Image address for pathout retrieval
_C.DATASET.ZEROSHOT_PO_IMDIR = None
# Pathout retrieval information 
_C.DATASET.ZEROSHOT_PO = None
# For csv-like datasets, which separator to use
_C.DATASET.CSV_SEPARATOR = '\t'
# For csv-like datasets, the name of the key for the image paths
_C.DATASET.CSV_IMG_KEY = 'image_name'
# For csv-like datasets, the name of the key for the captions
_C.DATASET.CSV_CAPTION_KEY = 'caption'
# Whether preload data
_C.DATASET.PRELOAD_DATA = False


# -----------------------------------------------------------------------------
# DATALODER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of dataloader workers per GPU
_C.DATALOADER.WORKORS = 8
# Batch size per GPU
_C.DATALOADER.BATCH_SIZE = 512

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# Number of epochs to train for
_C.SOLVER.EPOCHS = 100
# LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine
_C.SOLVER.LR_SCHEDULER = 'cosine'
# Epochs to freeze bert encoder
_C.SOLVER.FREE_BERT_EPOCHS = 0
# Epochs to freeze image encoder
_C.SOLVER.FREE_VISUAL_EPOCHS = 0
# Epochs to freeze knowledge encoder
_C.SOLVER.FREE_KNOWLEDGE_EPOCHS = 0
# Learning rate.
_C.SOLVER.LR = 1e-5
# End learning rate for cooldown schedule. Default: 0
_C.SOLVER.LR_COOLDOWN_END = 0.
# Power for polynomial cooldown schedule. Default: 1.0 (linear decay)
_C.SOLVER.LR_COOLDOWN_POWER = 1.
# Weight decay.
_C.SOLVER.WD = 0.2
# Number of steps to warmup for.
_C.SOLVER.WARMUP = 2000
#When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.
_C.SOLVER.EPOCHS_COOLDOWN = None
# How often to run evaluation with val data.
_C.SOLVER.VAL_FREQUENCY = 20
# How often to run zero shot.
_C.SOLVER.ZEROSHOT_FREQUENCY = 1
# Update the model every accum-freq steps.
_C.SOLVER.ACCUM_FREQ = 1 
# Log every n steps to tensorboard/console/wandb.
_C.SOLVER.LOG_EVERY_N_STEPS = 100

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Path to latest checkpoint (default: none)
_C.MODEL.RESUME = None
# Name of the vision backbone to use.
_C.MODEL.NAME = 'ViT-B-32'
# Use a pretrained CLIP model weights with the specified tag or file path. default is None

_C.MODEL.TEXT_ENCODER = 'bert'
_C.MODEL.BERT_PRETRAIN = None
# Use the knowledge encoder weights. defaut is none

_C.MODEL.IMAGE_ENCODER = 'clip_vit'
# Load imagenet pretrained weights for image tower backbone if available.
_C.MODEL.PRETRAINED_IMAGE = None

_C.MODEL.KNOWLEDGE_BERT = None
# Use the knowledge distillation. defaut is none
_C.MODEL.KNOWLEDGE_DISTILLATION = None
# Knowledge guidance
_C.MODEL.KNOWLEDGE_GUIDANCE = False
# Visual guidance
_C.MODEL.VISUAL_GUIDANCE = False
# Floating point precision ["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"]
_C.MODEL.PRECISION = 'amp'
# Gradient clip.
_C.MODEL.GRAD_CLIP_NORM = None

# TEXT embedding
_C.MODEL.TEXT_EMBED_DIM = 512
# Visual embedding head
_C.MODEL.VISUAL_EMBEDDING_HEAD = False
# Text embedding head
_C.MODEL.TEXT_EMBEDDING_HEAD = False
# Logit scale
_C.MODEL.LOGIT_SCALE = 0.07

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Loss weights for knowledge distillation
_C.LOSS.WEIGHT = [1.,1.,1.]

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
_C.SAVE = CN()
# Where to store tensorboard logs. Use None to avoid storing logs.
_C.SAVE.OUTPUT_IDR = './training/logs/'
# Optional identifier for the experiment when storing logs. Otherwise use current time.
_C.SAVE.NAME = 'default'
# How often to save checkpoints.
_C.SAVE.SAVE_FREQUENCY = 100
# Always save the most recent model (state_dict) trained to epoch_latest.bin.
_C.SAVE.SAVE_MOST_RECENT = True


# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()