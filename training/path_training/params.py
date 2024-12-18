import argparse
import ast


# def get_default_params(model_name):
#     # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
#     model_name = model_name.lower()
#     if "vit" in model_name:
#         return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
#     else:
#         return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--freeze-bert-epochs",
    #     default=0,
    #     type=int,
    #     help="Number of epochs for freezing text encoder in CLIP.",
    # )
    # parser.add_argument(
    #     "--name",
    #     type=str,
    #     default='clip_origin_pretrain_lr1',
    #     help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    # )
    # parser.add_argument(
    #     "--image-dir",
    #     type=str,
    #     default='../data/path_images_512/',
    #     help="Path to image dir",
    # )
    # parser.add_argument(
    #     "--preload-data",
    #     default=True,
    #     help="preload data",
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-XXX",  # RN50 by default
        help="Name of the vision backbone to use.",
    )
    # parser.add_argument(
    #     "--pretrained",   ## for openai clip 
    #     default='openai', #'../../pathclip_eval/model_zoo/ViT-B-32.pt',
    #     type=str,
    #     help="Use a pretrained CLIP model weights with the specified tag or file path. default is None",
    # )
    # parser.add_argument(
    #     "--bert-pretrain",
    #     default=None,#'../../A2_KEBERT/pretrained_model/pubmedbert',
    #     type=str,
    #     help="Use a pretrained bert weights. defaut is none",
    # )
    # parser.add_argument(
    #     "--knowledge-bert",
    #     default=None,#'../../A2_KEBERT/results/logs/pkbert_adasp_knowledge_encoder/checkpoints/epoch_500.pt',
    #     type=str,
    #     help="Use the knowledge encoder weights. defaut is none",
    # )
    # parser.add_argument(
    #     "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    # )
    # parser.add_argument(
    #     "--batch-size", type=int, default=512, help="Batch size per GPU."  # 64 by default
    # )
    # parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")  # None by default
    # parser.add_argument(
    #     "--lr-cooldown-end", type=float, default=1e-7,  # 0 by default
    #     help="End learning rate for cooldown schedule. Default: 0"
    # )
    # parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")  #0.2 by default
    # parser.add_argument(
    #     "--warmup", type=int, default=2000, help="Number of steps to warmup for."  #10000 by default
    # )
    # parser.add_argument(
    #     "--val-frequency", type=int, default=20, help="How often to run evaluation with val data."
    # )
    # parser.add_argument(
    #     "--zeroshot-frequency", type=int, default=1, help="How often to run zero shot."
    # )
    # parser.add_argument(
    #     "--save-best", type=bool, default=True, help="save the best model."
    # )
    # parser.add_argument(
    #     "--train-data",
    #     type=str,
    #     default='../data/merge_train_update1.csv',
    #     help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    # )
    # parser.add_argument(
    #     "--val-data",
    #     type=str,
    #     default='../data/merge_val_update1.csv',
    #     help="Path to file(s) with validation data",
    # )
    # parser.add_argument(
    #     "--zeroshot-cls-imdir",
    #     type=str,
    #     default='../../pathclip_eval/val_data/Kather/CRC-VAL-HE-7K',
    #     help="Path to file(s) with zeroshot classfication data",
    # )
    # parser.add_argument(
    #     "--zeroshot-cls",
    #     type=str,
    #     default='../../pathclip_eval/val_data/Kather/Kather_test.csv',
    #     help="Path to file(s) with zeroshot classification data, default is none",
    # )
    # parser.add_argument(
    #     "--zeroshot-ret-imdir",
    #     type=str,
    #     default='../../pathclip_eval/val_data/Arch_pubmed/arch_pubmed_retrieval',
    #     help="Path to file(s) with zeroshot retrieval data",
    # )
    # parser.add_argument(
    #     "--zeroshot-ret",
    #     type=str,
    #     default='../../pathclip_eval/val_data/Arch_pubmed/Arch_pubmed_test.csv',
    #     help="Path to file(s) with zeroshot validation data, default is none",
    # )
    # parser.add_argument(
    #     "--dataset-type",
    #     choices=["webdataset", "csv", "synthetic", "auto"],
    #     default="csv",
    #     help="Which type of dataset to process."
    # )
    # parser.add_argument(
    #     "--csv-separator",
    #     type=str,
    #     default="\t",
    #     help="For csv-like datasets, which separator to use."
    # )
    # parser.add_argument(
    #     "--csv-img-key",
    #     type=str,
    #     default="image_name",
    #     help="For csv-like datasets, the name of the key for the image paths."
    # )
    # parser.add_argument(
    #     "--csv-caption-key",
    #     type=str,
    #     default="caption",
    #     help="For csv-like datasets, the name of the key for the captions."
    # )
    # parser.add_argument(
    #     "--logs",
    #     type=str,
    #     default="./training/logs/",
    #     help="Where to store tensorboard logs. Use None to avoid storing logs.",
    # )
    # parser.add_argument(
    #     "--epochs", type=int, default=100, help="Number of epochs to train for."  #32 by default
    # )
    # parser.add_argument(
    #     "--epochs-cooldown", type=int, default=None,
    #     help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    # )
    # parser.add_argument(
    #     "--resume",
    #     default=None, #'./training/logs/PathKnowledge/checkpoints/epoch_68.pt',
    #     type=str,
    #     help="path to latest checkpoint (default: none)",
    # )
    # parser.add_argument(
    #     "--precision",
    #     choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
    #     default="amp",
    #     help="Floating point precision."
    # )
    # parser.add_argument(
    #     "--pretrained-image",
    #     default=False,
    #     action='store_true',
    #     help="Load imagenet pretrained weights for image tower backbone if available.",
    # )
    # parser.add_argument(
    #     "--lr-scheduler",
    #     type=str,
    #     default='cosine',
    #     help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    # )
    # parser.add_argument(
    #     "--lr-cooldown-power", type=float, default=1.0,
    #     help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    # )
    # parser.add_argument(
    #     "--save-frequency", type=int, default=20, help="How often to save checkpoints." # 1 by default
    # )
    # parser.add_argument(
    #     "--save-most-recent",
    #     action="store_true",
    #     default=True,
    #     help="Always save the most recent model (state_dict) trained to epoch_latest.bin.",
    # )
    # parser.add_argument(
    #     "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    # )
    # parser.add_argument(
    #     "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    # )

    ## no need to change
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=100669,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=12529,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='tensorboard', # '' by default
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
