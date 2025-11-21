#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import yaml
from datetime import datetime
import json
import logging
import os
from tqdm import tqdm
from collections import OrderedDict
from timm import utils

from utils.metrics import compute_metrics
from datasets import create_dataloader
import torch.nn.parallel
from models.mamba_vision import *
from timm.models import create_model, safe_model_name
from timm.utils import setup_default_logging, ParseKwargs

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')
_logger = logging.getLogger('validate')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
# Model parameters
# mamba_vision_T,mamba_vision_T2,mamba_vision_S,mamba_vision_B,mamba_vision_B_21k,mamba_vision_L,mamba_vision_L_21k
# mamba_vision_L2,mamba_vision_L2_512_21k

parser.add_argument('--model', default='mamba_vision_T', type=str, metavar='MODEL',help='Name of model to train (default: "gc_vit_tiny"')
parser.add_argument('--train_dataset_dir', metavar='DIR', default='./data/Cabbage6/train',help='path to train dataset')
parser.add_argument('--test_dataset_dir', metavar='DIR', default='./data/Cabbage6/test8',help='path to test dataset')

parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',help='use pre-trained model')
parser.add_argument('--resume', default='./resume', type=str, help='use best model')
parser.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment, name of sub-folder for output')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('--dataset_name', '-d', metavar='NAME', default='Cabbage6',help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--model_type', metavar='NAME', default='cabbage',help='')
parser.add_argument('--tag', default='exp', type=str, metavar='TAG')
parser.add_argument('--image_size', type=int, default=512, help='')

parser.add_argument('--use_random_crop', default=True, help='')
parser.add_argument('--use_random_horizontal_flip', default=True,help='')
parser.add_argument('--use_cutout', default=False,help='')
parser.add_argument('--use_random_erasing', default=False,help='')
parser.add_argument('--use_dual_cutout', default=False,help='')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=2, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', default=False,help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=6,help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,metavar='N', help='batch logging frequency (default: 10)')

parser.add_argument('--num-gpu', type=int, default=1,help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', help='enable test time pool')
parser.add_argument('--no-prefetcher', default=False,help='disable fast prefetcher')
parser.add_argument('--pin-mem', default=False,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', default=False,help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,help="Device (accelerator) to use.")
parser.add_argument('--amp', default=False,help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', default=False,help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, help='enable experimental fast-norm')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--retry', default=False, help='Enable batch size decay & retry for single model validation')


def validate(args):
    # might as well try to validate something
    # args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        **args.model_kwargs,
    )
    # move model to GPU, enable channels last layout if set
    model.to(device)
    args.resume = os.path.join(os.path.join(args.resume, args.model),'model_best.pth.tar')
    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))
    # 数据加载器
    _, loader_eval = create_dataloader(args, is_train=True)
    model.eval()

    with torch.no_grad():
        groundtruth = []
        preds = []
        for batch_idx, (input, target) in enumerate(tqdm(loader_eval)):
            input = input.to(device)
            target = target.to(device)
            groundtruth.append(target)
            output = model(input)
            # loss = loss_fn(output, target)
            # print(loss)
            _, pred = output.topk(1, 1, True, True)
            preds.append(pred)

        groundtruth = torch.cat(groundtruth)
        preds = torch.cat(preds).squeeze()
        acc1, precision, recall, f1, _, _ = compute_metrics(preds.cpu(), groundtruth.cpu())

        results = OrderedDict(
            acc1=round(acc1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )

    return results

def main():

    setup_default_logging()
    args = parser.parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.test_dataset_dir.split('/')[-1], safe_model_name(args.model)])
        args.experiment = exp_name

    results = validate(args)
    output_dir = utils.get_outdir(args.output if args.output else f'./output/validation/{args.tag}/', exp_name)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    args.results_file = os.path.join(output_dir, '.'.join(['result', args.results_format]))
    write_results(args.results_file, results, format=args.results_format)

def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()