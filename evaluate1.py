#!/usr/bin/env python

import argparse
import pathlib
import time
import torch
import gc

from pytorch_image_classification import (
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    compute_metrics,
)
from tqdm import tqdm


def load_config():
    parser = argparse.ArgumentParser()
    ## 'configs/cabbage/densenet.yaml' 中test_dataset_dir: ./data/Cabbage3/test0
    ## 'configs/cabbage/pyramidnet.yaml'
    ## 'configs/cabbage/resnet.yaml'
    ## 'configs/cabbage/resnet_preact.yaml'
    ## 'configs/cabbage/resnext.yaml'
    ## 'configs/cabbage/se_resnet_preact.yaml'
    ## 'configs/cabbage/shake_shake.yaml'
    ## 'configs/cabbage/vgg.yaml'
    ## 'configs/cabbage/wrn.yaml'
    parser.add_argument('--config', type=str, default='configs/cabbage/shake_shake.yaml')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)

    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)
    model.eval()
    loss_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        groundtruth = []
        preds = []
        for step, (data, targets) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            targets = targets.to(device)
            groundtruth.append(targets)
            output = model(data)
            _, pred = output.topk(1, 1, True, True)
            preds.append(pred)
            loss = loss_func(output, targets)
            loss_meter.update(loss.item())

        groundtruth = torch.cat(groundtruth)
        preds = torch.cat(preds)
        acc1, precision, recall, f1, _, _ = compute_metrics(preds.cpu(), groundtruth.cpu())

        logger.info(f'loss {loss_meter.avg:.4f} '
                    f'acc@1 {acc1:.4f} '
                    f'precision@1 {precision:.4f} '
                    f'recall@1 {recall:.4f} '
                    f'f1@1 {f1:.4f} ')

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')


def main():
    config = load_config()

    # 设置输出目录
    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # 日志文件名
    outfilename = '.'.join(['-'.join(['log', config.dataset.test_dataset_dir.split('/')[-1]]), 'txt'])
    logger = create_logger(name=__name__, output_dir=output_dir, filename=outfilename)

    # 创建模型
    model = create_model(config).to(config.device)

    # 加载 checkpoint 并过滤无关键
    if config.test.checkpoint != '':
        checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # 🔧 过滤掉所有包含 total_ops / total_params 的键
        state_dict = {k: v for k, v in state_dict.items()
                      if "total_ops" not in k and "total_params" not in k}
        model.load_state_dict(state_dict, strict=True)

    # dataloader & loss
    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    # 评估
    evaluate(config, model, test_loader, test_loss, logger)

    # 清理内存（防止循环时 OOM）
    del test_loader, test_loss
    torch.cuda.empty_cache()
    gc.collect()




if __name__ == '__main__':
    main()
