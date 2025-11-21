""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examplesf
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2023 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from tqdm import tqdm

from utils.metrics import compute_metrics

from datasets import create_dataloader
from timm.data import resolve_data_config
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
from timm.loss import BinaryCrossEntropy,LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from scheduler.scheduler_factory import create_scheduler
import shutil
from tensorboard import TensorboardLogger
from models.mamba_vision import *

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
group.add_argument('--train_dataset_dir', metavar='DIR', default='./data/Cabbage6/train',help='path to train dataset')
group.add_argument('--test_dataset_dir', metavar='DIR', default='./data/Cabbage6/test',help='path to test dataset')
group.add_argument('--dataset_name', '-d', metavar='NAME', default='Cabbage6',help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--model_type', metavar='NAME', default='cabbage',help='')
group.add_argument('--tag', default='exp', type=str, metavar='TAG')
group.add_argument('--image_size', type=int, default=512, help='')

# Model parameters
# mamba_vision_T,mamba_vision_T2,mamba_vision_S,mamba_vision_B,mamba_vision_B_21k,mamba_vision_L,mamba_vision_L_21k
# mamba_vision_L2,mamba_vision_L2_512_21k
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='mamba_vision_B_21k', type=str, metavar='MODEL',help='Name of model to train (default: "gc_vit_tiny"')
group.add_argument('--pretrained', default=False, action='store_true', help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--loadcheckpoint', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=6, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

group.add_argument('--smoothing', type=float, default=0.1,help='Label smoothing (default: 0.1)')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')
group.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=0.05,help='weight decay (default: 0.05)')
group.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',help='Clip gradient norm (default: 5.0, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',help='LR scheduler (default: "step"')
parser.add_argument('--lr-ep', action='store_true', default=False,help='using the epoch-based scheduler')
group.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='learning rate (default: 1e-3)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
group.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 310)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--use_random_crop', default=True, help='')
group.add_argument('--use_random_horizontal_flip', default=True,help='')
group.add_argument('--use_cutout', default=False,help='')
group.add_argument('--use_random_erasing', default=False,help='')
group.add_argument('--use_dual_cutout', default=False,help='')
group.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Drop of the attention, gaussian std')
group.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 3)')
group.add_argument('-j', '--workers', type=int, default=0, metavar='N',
                    help='how many training processes to use (default: 8)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--no-ddp-bb', action='store_true', default=False, help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment, name of sub-folder for output')
group.add_argument('--log_dir', default='./log_dir/', type=str, help='where to store tensorboard')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC', help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N', help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument("--data_len", default=2015, type=int,help='size of the dataset')

group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')

group.add_argument('--no_saver', action='store_true', default=False, help='Save checkpoints')
group.add_argument('--ampere_sparsity', action='store_true', default=False,
                    help='Save checkpoints')
group.add_argument('--bfloat', action='store_true', default=False,
                    help='use bfloat datatype')
group.add_argument('--mesa',  type=float, default=0.0,
                    help='use memory efficient sharpness optimization, enabled if >0.0')
group.add_argument('--mesa-start-ratio',  type=float, default=0.25,
                    help='when to start MESA, ratio to total training time, def 0.25')

kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

def freeze_parameters(model, params_to_freeze):
    for name, param in model.named_parameters():
        if any(name.startswith(ptf) for ptf in params_to_freeze):
            param.requires_grad = False

def kdloss(y, teacher_scores):
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    args.rank = 0
    args.world_size = 1
    args.device = 'cuda:0'
    if not torch.cuda.is_available():
        args.device = 'cpu'

    utils.random_seed(args.seed)
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        attn_drop_rate=args.attn_drop_rate,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path)
    
    if args.bfloat:
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float16

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # move model to GPU, enable channels last layout if set
    model.to(args.device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    print("filter_bias_and_bn")
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    loss_scaler = None
    # optionally resume from a checkpoint
    resume_epoch = None

    if not os.path.isfile(args.resume):
        args.resume = ""

    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = utils.ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.loadcheckpoint:
        _logger.info(r"Loading checkpoint {args.loadcheckpoint}, checking for existing parameters if their shape match")
        # 修改加载检查点的方式，仅加载权重
        new_model_weights = torch.load(args.loadcheckpoint, map_location=args.device, weights_only=True)
        current_model = model.state_dict()
        new_state_dict = OrderedDict()
        for k in current_model.keys():
            if k in new_model_weights.keys():
                if new_model_weights[k].size() == current_model[k].size():
                    print(r"loading weights {k} {new_model_weights[k].size()}")
                    new_state_dict[k] = new_model_weights[k]
        model.load_state_dict(new_state_dict, strict=False)
        if model_ema is not None:
            model_ema.module.load_state_dict(new_state_dict, strict=False)

   # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # 数据加载器
    loader_train, loader_eval = create_dataloader(args, is_train=True)

     # setup loss function
    if args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(args.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        log_dir = args.log_dir + '_' + args.tag
        os.makedirs(log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=log_dir)
    else:
        log_writer = None

    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), safe_model_name(args.model),str(data_config['input_size'][-1])])
            args.experiment = exp_name

        output_dir = utils.get_outdir(args.output if args.output else f'./output/train/{args.tag}/', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=1)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        # if 1: #args.copy_code
        #     # copy .py files
        #     files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if
        #              os.path.splitext(f)[1] == '.py']
        #     for f in files:
        #         if "/code_copy/" in f: continue
        #         new_path = output_dir + "/code_copy/" + f
        #         os.makedirs(os.path.dirname(new_path), exist_ok=True)
        #         shutil.copyfile(f, new_path)

    test_acc_track=[]
    try:
        # 冻结部分参数
        params_to_freeze = ['patch_embed','levels.0','levels.1','levels.2']
        freeze_parameters(model, params_to_freeze)

        for epoch in range(start_epoch, num_epochs):
            saver = saver if not args.no_saver else None
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                loss_scaler=loss_scaler, model_ema=model_ema)

            eval_metrics = validate(model, epoch, loader_eval, validate_loss_fn, args)
            if log_writer is not None:
                log_writer.update(test_acc1=eval_metrics['top1'], head="perf", step=epoch)
                log_writer.update(test_loss=eval_metrics['loss'], head="perf", step=epoch)
                log_writer.update(train_loss=train_metrics['loss'], head="perf", step=epoch)
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                log_writer.update(lr=lr, head="perf", step=epoch)

            test_acc_track.append(eval_metrics['top1'])
            stopif = True if len(test_acc_track)>1 and test_acc_track[-1]<1.0 else False

            if log_writer is not None:
                log_writer.update(test_acc1_ema=eval_metrics['top1'], head="perf", step=epoch)
                log_writer.update(test_loss_ema=eval_metrics['loss'], head="perf", step=epoch)

            if lr_scheduler is not None and args.lr_ep:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, None if eval_metrics is None else eval_metrics[eval_metric])

            if output_dir is not None:
                utils.update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = None if eval_metrics is None else eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            # if not np.isfinite(eval_metrics['loss']) or stopif:
            #     # if got None then exit
            #     if args.local_rank == 0:
            #         _logger.info("Nan in loss, exit")
            #         _logger.error("Nan in loss, exit")
            #
            #     exit(1)
            #     return 0

    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None,
        loss_scaler=None, model_ema=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    num_iters = len(loader)
    device = torch.device(args.device)

    for batch_idx, (input, target) in enumerate(tqdm(loader)):

        if lr_scheduler is not None and not args.lr_ep:
            lr_scheduler.step_update(num_updates=(epoch * num_iters) + batch_idx + 1)

        if (batch_idx == 0) or (batch_idx % 50 == 0):
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = loss_fn(output, target)
        # print(loss)
        if args.mesa > 0.0:
            if epoch / args.epochs > args.mesa_start_ratio:
                with torch.no_grad():
                    ema_output = model_ema.module(input).data.detach()
                kd = kdloss(output, ema_output)
                loss += args.mesa * kd

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward(create_graph=second_order)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:

            if args.clip_grad is not None:
                utils.dispatch_clip_grad(model_parameters(model, exclude_head='agc' in args.clip_mode),value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:

            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and args.lr_ep:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, epoch, loader, loss_fn, args, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    device = torch.device(args.device)
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        groundtruth = []
        preds = []
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            last_batch = batch_idx == last_idx

            input = input.to(device)
            target = target.to(device)
            groundtruth.append(target)

            output = model(input)
            loss = loss_fn(output, target)
            # print(loss)
            _, pred = output.topk(1, 1, True, True)
            preds.append(pred)
            reduced_loss = loss.data
            losses_m.update(reduced_loss.item(), input.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m))

        groundtruth = torch.cat(groundtruth)
        preds = torch.cat(preds).squeeze()
        acc1, precision, recall, f1, _, _ = compute_metrics(preds.cpu(), groundtruth.cpu())

        _logger.info(f'Epoch {epoch} '
                    f'acc@1 {acc1:.4f} '
                    f'precision@1 {precision:.4f} '
                    f'recall@1 {recall:.4f} '
                    f'f1@1 {f1:.4f} '
                    )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', acc1)])
    return metrics

if __name__ == '__main__':
    main()
