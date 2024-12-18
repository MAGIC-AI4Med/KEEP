import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
try:
    import wandb
except ImportError:
    wandb = None
try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

sys.path.append("./")
from path_open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from path_training.data_proc import get_data, preload_dataset
from path_training.distributed import is_master, init_distributed_device, broadcast_object
from path_training.logger import setup_logging
from path_training.params import parse_args
from path_training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from path_training.train import train_one_epoch, evaluate
from path_training.file_utils import pt_load, start_sync_process, remote_sync
from path_training.config import cfg
from path_training.freeze_scheduler import FreezeScheduler, FreezeChecker
import argparse

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(input_args):
    parser = argparse.ArgumentParser(description="CLIP Training")
    parser.add_argument(
        "--config_file", default="./configs/KEP-32_OpenPath_example.yml", help="path to config file", type=str
    )
    init_args = parser.parse_args()
    args = parse_args(input_args)
    
    if init_args.config_file != "":
        cfg.merge_from_file(init_args.config_file)
    # cfg.freeze()

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if cfg.SAVE.NAME is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = cfg.MODEL.NAME.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        cfg.MODEL.NAME = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{cfg.SOLVER.LR}",
            f"b_{cfg.DATALOADER.BATCH_SIZE}",
            f"j_{cfg.DATALOADER.WORKORS}",
            f"p_{cfg.MODEL.PRECISION}",
        ])

    resume_latest = cfg.MODEL.RESUME == 'latest'
    log_base_path = os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, cfg.SAVE.NAME, "checkpoints")
            if cfg.SAVE.SAVE_MOST_RECENT:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if cfg.SAVE.SAVE_MOST_RECENT:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        cfg.MODEL.RESUME = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME), 
            os.path.join(args.remote_sync, cfg.SAVE.NAME), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME), 
            os.path.join(args.remote_sync, cfg.SAVE.NAME), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if cfg.MODEL.PRECISION == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')


    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        cfg.MODEL.NAME,
        cfg.MODEL.LOGIT_SCALE,
        cfg.MODEL.TEXT_ENCODER,
        cfg.MODEL.BERT_PRETRAIN,
        cfg.MODEL.IMAGE_ENCODER,
        cfg.MODEL.PRETRAINED_IMAGE,
        cfg.MODEL.KNOWLEDGE_BERT,
        cfg.MODEL.KNOWLEDGE_DISTILLATION,
        cfg.MODEL.VISUAL_EMBEDDING_HEAD,
        cfg.MODEL.TEXT_EMBEDDING_HEAD,
        precision=cfg.MODEL.PRECISION,
        device=device,
        jit=args.torchscript,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )

    random_seed(args.seed, 0)
    if args.trace:
        model = trace_model(model, batch_size=cfg.DATALOADER.BATCH_SIZE, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
    # get tokenizer
    tokenizer=get_tokenizer(cfg.MODEL.NAME, cfg.MODEL.BERT_PRETRAIN, cfg.MODEL.TEXT_ENCODER, cfg.MODEL.KNOWLEDGE_GUIDANCE)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    logging.info(f'Proccessing config file: {init_args.config_file}')
    if cfg.DATASET.TRAIN_DATA or cfg.DATASET.TYPE == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": cfg.SOLVER.WD},
            ],
            lr=cfg.SOLVER.LR,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if cfg.MODEL.PRECISION == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if cfg.MODEL.RESUME is not None:
        checkpoint = pt_load(cfg.MODEL.RESUME, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            
            logging.info(f"=> resuming checkpoint '{cfg.MODEL.RESUME}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{cfg.MODEL.RESUME}' (epoch {start_epoch})")

    # preload all_data
    if cfg.DATASET.PRELOAD_DATA == True:
        all_data = preload_dataset(cfg)
        args.preload_data = all_data
    else:
        args.preload_data = None

    # initialize datasets
    data = get_data(args, cfg, (preprocess_train, preprocess_val))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // cfg.SOLVER.ACCUM_FREQ) * cfg.SOLVER.EPOCHS
        if cfg.SOLVER.LR_SCHEDULER == "cosine":
            scheduler = cosine_lr(optimizer, cfg.SOLVER.LR, cfg.SOLVER.WARMUP, total_steps)
        elif cfg.SOLVER.LR_SCHEDULER == "const":
            scheduler = const_lr(optimizer, cfg.SOLVER.LR, cfg.SOLVER.WARMUP, total_steps)
        elif cfg.SOLVER.LR_SCHEDULER == "const-cooldown":
            assert cfg.SOLVER.EPOCHS_COOLDOWN is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // cfg.SOLVER.ACCUM_FREQ) * cfg.SOLVER.EPOCHS_COOLDOWN
            scheduler = const_lr_cooldown(
                optimizer, cfg.SOLVER.LR, cfg.SOLVER.WARMUP, total_steps,
                cooldown_steps, cfg.SOLVER.SOLVER.LR_COOLDOWN_POWER, cfg.SOLVER.LR_COOLDOWN_END)
        else:
            logging.error(
                f'Unknown scheduler, {cfg.SOLVER.LR_SCHEDULER}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = cfg.SAVE.OUTPUT_IDR and cfg.SAVE.OUTPUT_IDR.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=cfg.SAVE.NAME,
            id=cfg.SAVE.NAME,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if cfg.MODEL.RESUME == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from path_open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, tokenizer, start_epoch, args, cfg, writer)
        return

    loss = create_loss(args,cfg)

    # if cfg.MODEL.RESUME is not None:
    #     test_results = evaluate(model, data, tokenizer, start_epoch+1, args, cfg, writer)
    #     all_test_results = [test_results['zeroshot-cls-WF1-median']]
    # else:
    #     all_test_results = [-1]
    # print('best_wF1 so far: %.3f'%(max(all_test_results)))
    
    freezescheduler = FreezeScheduler()
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):

        logging.info('----------++++++++++++++++++++++++++-------------')

        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        ## freeze model layers
        freezescheduler.update(cfg, model, epoch)

        ## check state of model layers
        freezechecker = FreezeChecker(model)
        
        train_one_epoch(model, data, tokenizer, loss, epoch, optimizer, scaler, scheduler, args, cfg, tb_writer=writer)
        completed_epoch = epoch + 1

        freezechecker.freeze_checker(model)

        if any(v in data for v in ('val', 'zeroshot_cls', 'zeroshot_ret')):
            test_results = evaluate(model, data, tokenizer, completed_epoch, args, cfg, writer)

        # if 'zeroshot-cls-WF1-median' in test_results:
        #     if test_results['zeroshot-cls-WF1-median'] > max(all_test_results) and cfg.SAVE.SAVE_BEST:
        #         checkpoint_dict = {
        #         "epoch": completed_epoch,
        #         "name": cfg.SAVE.NAME,
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         }
        #         if scaler is not None:
        #             checkpoint_dict["scaler"] = scaler.state_dict()
        #             torch.save(
        #                 checkpoint_dict,
        #                 os.path.join(args.checkpoint_path, "epoch_best.pt"),
        #             )
        #     all_test_results.append(test_results['zeroshot-cls-WF1-median'])

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": cfg.SAVE.NAME,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == cfg.SOLVER.EPOCHS or (
                cfg.SAVE.SAVE_FREQUENCY  > 0 and (completed_epoch % cfg.SAVE.SAVE_FREQUENCY ) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if cfg.SAVE.SAVE_MOST_RECENT:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME), 
            os.path.join(args.remote_sync, cfg.SAVE.NAME), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(cfg.SAVE.OUTPUT_IDR, cfg.SAVE.NAME, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
