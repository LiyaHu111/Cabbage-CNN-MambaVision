import argparse
import os.path
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_image_classification.config.config_node import ConfigNode
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    compute_metrics,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    save_config,
    set_seed,
    setup_cudnn,
)

global_step = 0


def load_config():
    parser = argparse.ArgumentParser()
    ## 'configs/cabbage/densenet.yaml'
    ## 'configs/cabbage/pyramidnet.yaml'
    ## 'configs/cabbage/resnet.yaml'
    ## 'configs/cabbage/resnet_preact.yaml'
    ## 'configs/cabbage/resnext.yaml'
    ## 'configs/cabbage/se_resnet_preact.yaml'``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
    ## 'configs/cabbage/shake_shake.yaml'
    ## 'configs/cabbage/vgg.yaml'
    ## 'configs/cabbage/wrn.yaml'
    parser.add_argument('--config', type=str, default='configs/cabbage/shake_shake.yaml')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config


def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader, logger, tensorboard_writer, tensorboard_writer2):
    global global_step
    logger.info(f'Train {epoch} {global_step}')
    device = torch.device(config.device)
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(tqdm(train_loader)):
        step += 1
        global_step += 1
        if step == 1:
            if config.tensorboard.train_images:
                image = torchvision.utils.make_grid(data,normalize=True,scale_each=True)
                tensorboard_writer.add_image('Train/Image', image, epoch)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, targets)
        _, pred = output.topk(1, 1, True, True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        optimizer.step()
        acc1 = compute_accuracy(pred.cpu(), targets.cpu())
        loss_meter.update(loss.item())
        acc1_meter.update(acc1)

        if step % config.train.log_period == 0 or step == len(train_loader):
            logger.info(
                f'Epoch {epoch} '
                f'Step {step}/{len(train_loader)} '
                f'lr {scheduler.get_last_lr()[0]:.6f} '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) ')

            tensorboard_writer2.add_scalar('Train/RunningLoss', loss_meter.avg, global_step)
            tensorboard_writer2.add_scalar('Train/RunningAcc1', acc1_meter.avg, global_step)
            tensorboard_writer2.add_scalar('Train/RunningLearningRate', scheduler.get_last_lr()[0], global_step)

        scheduler.step()

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
    tensorboard_writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)


def validate(epoch, config, model, loss_func, val_loader, logger, tensorboard_writer):
    logger.info(f'Val {epoch}')
    device = torch.device(config.device)
    model.eval()
    loss_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        groundtruth = []
        preds = []
        for step, (data, targets) in enumerate(tqdm(val_loader)):
            if config.tensorboard.val_images:
                if epoch == 0 and step == 0:
                    image = torchvision.utils.make_grid(data, normalize=True, scale_each=True)
                    tensorboard_writer.add_image('Val/Image', image, epoch)
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

        logger.info(f'Epoch {epoch} '
                    f'loss {loss_meter.avg:.4f} '
                    f'acc@1 {acc1:.4f} '
                    f'precision@1 {precision:.4f} '
                    f'recall@1 {recall:.4f} '
                    f'f1@1 {f1:.4f} '
                    )

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

    if epoch > 0:
        tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Val/Acc1', acc1, epoch)
    tensorboard_writer.add_scalar('Val/precision', precision, epoch)
    tensorboard_writer.add_scalar('Val/recall', recall, epoch)
    tensorboard_writer.add_scalar('Val/f1', f1, epoch)
    tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)

    return acc1, precision, recall, f1


def main():
    global global_step
    config = load_config()
    set_seed(config)
    setup_cudnn(config)
    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2, size=config.scheduler.epochs)

    output_dir = pathlib.Path(config.train.output_dir)
    if not config.train.resume and output_dir.exists():
        raise RuntimeError(f'Output directory `{output_dir.as_posix()}` already exists')
    output_dir.mkdir(exist_ok=True, parents=True)
    if not config.train.resume:
        save_config(config, output_dir / 'config.yaml')
        save_config(get_env_info(config), output_dir / 'env.yaml')
        diff = find_config_diff(config)
        if diff is not None:
            save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,output_dir=output_dir,filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    # 数据加载器
    train_loader, val_loader = create_dataloader(config, is_train=True)

    model = create_model(config)
    macs, n_params = count_op(config, model)
    logger.info(f'MACs   : {macs}')
    logger.info(f'#params: {n_params}')

    optimizer = create_optimizer(config, model)

    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch

    best_acc = 0

    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()
    elif config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    if config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(config, output_dir, purge_step=config.train.start_epoch + 1)
        tensorboard_writer2 = create_tensorboard_writer(config, output_dir / 'running', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()

    train_loss, val_loss = create_loss(config)

    # if (config.train.val_period > 0 and start_epoch == 0 and config.train.val_first):
    #     validate(0, config, model, val_loss, val_loader, logger, tensorboard_writer)

    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1
        np.random.seed(seed)
        train(epoch, config, model, optimizer, scheduler, train_loss, train_loader, logger, tensorboard_writer, tensorboard_writer2)

        if config.train.val_period > 0 and (epoch % config.train.val_period == 0):
            acc1, _, _, _ = validate(epoch, config, model, val_loss, val_loader, logger, tensorboard_writer)
            if acc1 > best_acc:
                best_acc = acc1
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config.as_dict(),
                    'state_dict': model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(output_dir,'best_model.pth'))

        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        if (epoch % config.train.checkpoint_period == 0) or (epoch == config.scheduler.epochs):
            checkpoint_config = {
                'epoch': epoch,
                'global_step': global_step,
                'config': config.as_dict(),
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint_config, os.path.join(output_dir,f'checkpoint_{epoch:05d}.pth'))

    tensorboard_writer.close()
    tensorboard_writer2.close()

if __name__ == '__main__':
    main()
