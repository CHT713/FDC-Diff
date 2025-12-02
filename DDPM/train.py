
import os
import torch


import logging
import random
import shutil
import sys
import time
from datetime import datetime

from colorama import Fore, Style
from easydict import EasyDict
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset

from datasets import *
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import yaml
import argparse
from tqdm import tqdm
from get_model import get_model




# 解析参数保持不变
p = argparse.ArgumentParser()
p.add_argument('--config', type=str, default=None)
p.add_argument('--data', action='store', type=str, default="datasets")
p.add_argument('--train_data_prefix', action='store', type=str, default='train')
p.add_argument('--val_data_prefix', action='store', type=str, default='val')
p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
p.add_argument('--logs', action='store', type=str, default='logs')
p.add_argument('--device', action='store', type=str, default='cuda:0')
p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
p.add_argument('--log_iterations', action='store', type=str, default=20)
p.add_argument('--exp_name', type=str, default='YourName')
p.add_argument('--model', type=str, default='egnn_dynamics',
               help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')
p.add_argument('--diffusion_steps', type=int, default=1000)
p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
p.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')
p.add_argument('--n_epochs', type=int, default=100)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--brute_force', type=eval, default=False, help='True | False')
p.add_argument('--actnorm', type=eval, default=True, help='True | False')
p.add_argument('--break_train_epoch', type=eval, default=False, help='True | False')
p.add_argument('--dp', type=eval, default=True, help='True | False')
p.add_argument('--condition_time', type=eval, default=True, help='True | False')
p.add_argument('--clip_grad', type=eval, default=True, help='True | False')
p.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
p.add_argument('--n_layers', type=int, default=6, help='number of layers')
p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
p.add_argument('--nf', type=int, default=128, help='number of layers')
p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
p.add_argument('--norm_constant', type=float, default=1, help='diff/(|diff| + norm_constant)')
p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
p.add_argument('--ode_regularization', type=float, default=1e-3)
p.add_argument('--dataset', type=str, default='crossdock', help='crossdock')
p.add_argument('--datadir', type=str, default='/crossdock/', help='crossdock directory')
p.add_argument('--filter_n_atoms', type=int, default=None, help='')
p.add_argument('--dequantization', type=str, default='argmax_variational',
               help='uniform | variational | argmax_variational | deterministic')
p.add_argument('--n_report_steps', type=int, default=1)
p.add_argument('--wandb_usr', type=str)
p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
p.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
p.add_argument('--save_model', type=eval, default=True, help='save model')
p.add_argument('--generate_epochs', type=int, default=1, help='save model')
p.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
p.add_argument('--test_epochs', type=int, default=1)
p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
p.add_argument('--resume', type=str, default=None, help='')
p.add_argument('--start_epoch', type=int, default=0, help='')
p.add_argument('--ema_decay', type=float, default=0.999,
               help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
p.add_argument('--augment_noise', type=float, default=0)
p.add_argument('--n_stability_samples', type=int, default=500, help='Number of samples to compute the stability')
p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
               help='normalize factors for [x, categorical, integer]')
p.add_argument('--remove_h', action='store_true')
p.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
p.add_argument('--visualize_every_batch', type=int, default=1e8,
               help="Can be used to visualize multiple times per epoch")
p.add_argument('--seed', type=int, default=None, help='随机种子，若不指定则随机生成')

p.add_argument('--normalization_factor', type=float, default=1, help="Normalize the sum aggregation of EGNN")
p.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')

p.add_argument('--normalization', type=str, default='batch_norm', help='normalization type (batch_norm, layer_norm)')

p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
p.add_argument('--center_of_mass', type=str, default='scaffold', help='Where to center the data: scaffold | anchors')
p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')
p.add_argument('--cuda', type=bool, default=True)
p.add_argument('--logdir', type=str, default='./logs')
p.add_argument('--resume_checkpoint', type=str, default=None, help="断点恢复的检查点路径")


# 训练阶段函数
def train(it, model, optimizers, train_loader, logger, writer, current_epoch, scaffold_epochs, total_epoch, device):
    model.train()
    os.environ["PYTHONUNBUFFERED"] = "1"  # 确保实时输出

    sum_scaffold_loss = 0
    sum_rgroup_loss = 0
    sum_n = 0
    skipped_batches = 0

    # 解包优化器
    optimizer_scaffold, optimizer_rgroup = optimizers

    # 始终使用两个优化器
    active_optimizers = [optimizer_scaffold, optimizer_rgroup]

    with tqdm(total=len(train_loader), desc=f'Training Epoch {it}', file=sys.stdout,
              leave=True, mininterval=0.1, ascii=True, unit='batch', dynamic_ncols=True) as pbar:
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # 检查输入数据是否含有 NaN/Inf
            skip_flag = False
            for k, v in batch.items():
                if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                    logger.warning(f"Epoch {current_epoch}, Batch {batch_idx} - 跳过输入中存在 NaN/Inf 的键: {k}")
                    skipped_batches += 1
                    pbar.update(1)
                    skip_flag = True
                    break
            if skip_flag:
                continue

            # 调用 model.forward，始终返回骨架和 R 基团的损失
            scaffold_loss, rgroup_loss = model.forward(
                batch, training=True, current_epoch=current_epoch,
                scaffold_epochs=scaffold_epochs, total_epoch=total_epoch
            )
            balanced_scaffold = scaffold_loss if scaffold_loss is not None else None
            balanced_rgroup = rgroup_loss  if rgroup_loss is not None else None

            # 清零优化器梯度
            for opt in active_optimizers:
                opt.zero_grad(set_to_none=True)

            # 处理骨架部分的损失
            if balanced_scaffold is not None and not (
                    torch.isnan(balanced_scaffold).any() or torch.isinf(balanced_scaffold).any()):
                balanced_scaffold.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_scaffold.step()  # 更新参数
            else:
                logger.warning(f"Epoch {current_epoch}, Batch {batch_idx} - Invalid scaffold loss, skipping")
                pbar.set_postfix({'status': 'Invalid scaffold loss'})
                optimizer_scaffold.zero_grad(set_to_none=True)

            # 处理 R 基团部分的损失
            if balanced_rgroup is not None and not (
                    torch.isnan(balanced_rgroup).any() or torch.isinf(balanced_rgroup).any()):
                balanced_rgroup.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_rgroup.step()  # 更新参数
            else:
                logger.warning(f"Epoch {current_epoch}, Batch {batch_idx} - Invalid rgroup loss, skipping")
                pbar.set_postfix({'status': 'Invalid rgroup loss'})
                optimizer_rgroup.zero_grad(set_to_none=True)

            # 计算批次大小
            first_key = list(batch.keys())[0]
            first_value = batch[first_key]
            if torch.is_tensor(first_value):
                current_batch_size = first_value.size(0)
            elif isinstance(first_value, list):
                current_batch_size = len(first_value)
            else:
                logger.error(
                    f"Epoch {current_epoch}, Batch {batch_idx} - Unsupported type for {first_key}: {type(first_value)}")
                raise ValueError(f"Unsupported batch value type: {type(first_value)}")

            # 记录损失
            scaffold_loss_item = balanced_scaffold.item() if balanced_scaffold is not None else 0
            rgroup_loss_item = balanced_rgroup.item() if balanced_rgroup is not None else 0
            sum_scaffold_loss += scaffold_loss_item * current_batch_size
            sum_rgroup_loss += rgroup_loss_item * current_batch_size
            sum_n += current_batch_size

            pbar.set_postfix({
                'scaffold': f'{scaffold_loss_item:.2f}',
                'rgroup': f'{rgroup_loss_item:.2f}',
                'bs': current_batch_size
            })
            pbar.update(1)

    # 计算平均损失
    avg_scaffold_loss = sum_scaffold_loss / sum_n if sum_n > 0 else float('inf')
    avg_rgroup_loss = sum_rgroup_loss / sum_n if sum_n > 0 else float('inf')
    avg_total_loss = avg_scaffold_loss + avg_rgroup_loss
    logger.info(
        f"Training Epoch {it}, Avg Scaffold Loss: {avg_scaffold_loss:.4f}, Avg Rgroup Loss: {avg_rgroup_loss:.4f}, "
        f"Avg Total Loss: {avg_total_loss:.4f}, Skipped Batches: {skipped_batches}")
    torch.cuda.empty_cache()

    # 记录到 TensorBoard
    writer.add_scalar('train/scaffold_loss_epoch', avg_scaffold_loss, it)
    writer.add_scalar('train/rgroup_loss_epoch', avg_rgroup_loss, it)
    writer.add_scalar('train/total_loss_epoch', avg_total_loss, it)
    return avg_total_loss


# Validate function (unchanged except for parameter name consistency)
def validate(it, model, val_loader, logger, writer, scaffold_epochs, transition_epoch, total_epoch, device):
    model.eval()
    sum_scaffold_loss = 0.0
    sum_rgroup_loss = 0.0
    sum_total_loss = 0.0
    sum_n = 0
    skipped_batches = 0

    # 始终使用联合训练模式
    stage = 'joint'

    is_redirected = not sys.stdout.isatty()

    with tqdm(total=len(val_loader), desc=f'Validation Epoch {it} Stage={stage}', disable=is_redirected,
              file=sys.stdout,
              leave=True, mininterval=0.1, ascii=True, unit='batch', dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # 调用 model.forward，始终返回骨架和 R 基团的损失
                scaffold_loss, rgroup_loss = model.forward(
                    batch, training=False, current_epoch=it,
                    scaffold_epochs=scaffold_epochs, total_epoch=total_epoch
                )

                balanced_scaffold = scaffold_loss if scaffold_loss is not None else None
                balanced_rgroup = rgroup_loss if rgroup_loss is not None else None

                # 始终检查两部分损失的有效性
                if (balanced_scaffold is None or torch.isnan(balanced_scaffold).any() or torch.isinf(
                        balanced_scaffold).any()) or \
                        (balanced_rgroup is None or torch.isnan(balanced_rgroup).any() or torch.isinf(
                            balanced_rgroup).any()):
                    logger.warning(f"Epoch {it}, Batch {batch_idx} - Skipping validation batch (loss invalid)")
                    skipped_batches += 1
                    pbar.update(1)
                    continue

                scaffold_loss_item = balanced_scaffold.mean().item() if balanced_scaffold.ndim > 0 else balanced_scaffold.item()
                rgroup_loss_item = balanced_rgroup.mean().item() if balanced_rgroup.ndim > 0 else balanced_rgroup.item()

                try:
                    first_key = list(batch.keys())[0]
                    first_value = batch[first_key]
                    if torch.is_tensor(first_value):
                        batch_size = first_value.size(0)
                    elif isinstance(first_value, list):
                        batch_size = len(first_value)
                    else:
                        logger.error(
                            f"Epoch {it}, Batch {batch_idx} - Unsupported type for {first_key}: {type(first_value)}")
                        raise ValueError(f"Unsupported batch value type: {type(first_value)}")
                except Exception as e:
                    logger.warning(
                        f"Epoch {it}, Batch {batch_idx} - Failed to determine batch size: {e}, defaulting to 1")
                    batch_size = 1

                sum_scaffold_loss += scaffold_loss_item * batch_size
                sum_rgroup_loss += rgroup_loss_item * batch_size
                total_loss_item = scaffold_loss_item + rgroup_loss_item
                sum_total_loss += total_loss_item * batch_size
                sum_n += batch_size

                pbar.set_postfix({
                    'scaffold': f'{scaffold_loss_item:.4f}',
                    'rgroup': f'{rgroup_loss_item:.4f}',
                    'total': f'{total_loss_item:.4f}',
                    'bs': batch_size
                })
                pbar.update(1)

    avg_scaffold_loss = sum_scaffold_loss / sum_n if sum_n > 0 else float('inf')
    avg_rgroup_loss = sum_rgroup_loss / sum_n if sum_n > 0 else float('inf')
    avg_total_loss = sum_total_loss / sum_n if sum_n > 0 else float('inf')

    logger.info(f'Validation Epoch {it}, Stage: {stage}, Avg Scaffold Loss: {avg_scaffold_loss:.4f}, '
                f'Avg Rgroup Loss: {avg_rgroup_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}, '
                f'Skipped Batches: {skipped_batches}')
    writer.add_scalar('validation/scaffold_loss', avg_scaffold_loss, it)
    writer.add_scalar('validation/rgroup_loss', avg_rgroup_loss, it)
    writer.add_scalar('validation/total_loss', avg_total_loss, it)

    return avg_scaffold_loss, avg_rgroup_loss


# 工具函数保持不变
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        lr = float(cfg.lr)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                                betas=(cfg.beta1, cfg.beta2))
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    args = p.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda:0")

    config_path = args.config
    args.dataset = 'crossdock'

    # 生成随机种子或使用用户指定的种子
    if args.seed is None:
        # 使用当前时间的微秒部分生成随机种子
        seed = int((datetime.now().timestamp() % 1) * 1000000)
    else:
        seed = args.seed

    # 设置随机种子
    seed_all(seed)

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config.diffusion_steps = args.diffusion_steps
    config.diffusion_noise_schedule = args.diffusion_noise_schedule
    config.diffusion_noise_precision = args.diffusion_noise_precision
    config.diffusion_loss_type = args.diffusion_loss_type

    config_name = '{}_exp'.format(args.dataset)

    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)

    # 保存随机种子信息
    logger.info(f"使用的随机种子: {seed}")
    with open(os.path.join(log_dir, 'random_seed.txt'), 'w') as f:
        f.write(f"Random Seed: {seed}\n")

    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # 数据加载
    dataset_type = CrossDockDataset
    train_dataset = dataset_type(data_path=config.data_path, prefix=config.train_data_prefix, device="cpu")

    # 使用相同的随机种子初始化KFold，确保可复现
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    total_epoch = config.n_epochs

    # 模型和优化器
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"========== Fold {fold + 1} ==========")
        logger.info(f"Starting Fold {fold + 1}")

        # 创建训练和验证子集
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        # 创建数据加载器
        train_loader = get_dataloader(train_subset, config.batch_size, collate_fn=collate, shuffle=True, num_workers=0)
        val_loader = get_dataloader(val_subset, config.batch_size, collate_fn=collate, shuffle=False, num_workers=0)

        # 初始化模型、优化器和调度器
        model = get_model(config, device).to(device)
        optimizers, _ = model.configure_optimizers()
        schedulers = [get_scheduler(config.train.scheduler, opt) for opt in optimizers]
        scaffold_epochs = None

        best_scaffold_loss = float('inf')  # 骨架最佳损失
        best_rgroup_loss = float('inf')  # R 基团最佳损失
        avg_val_loss = float('inf')  # 联合最佳损失

        # 记录初始学习率
        for i, optimizer in enumerate(optimizers):
            initial_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Fold {fold + 1}, Optimizer {i} Initial LR: {initial_lr}")
            writer.add_scalar(f'learning_rate/fold_{fold + 1}/optimizer_{i}', initial_lr, 0)

        # 训练循环
        for it in range(1, total_epoch + 1):
            current_epoch = it
            print(Fore.YELLOW + f"========== Fold {fold + 1} Epoch {it} ==========" + Style.RESET_ALL)

            # 训练
            try:
                start_time = time.time()
                avg_train_loss = train(it, model, optimizers, train_loader, logger, writer,
                                       current_epoch, scaffold_epochs, total_epoch, device)
                if avg_train_loss is None:
                    print(Fore.YELLOW + f"Fold {fold + 1} Epoch {it} Training Failed, Skipping" + Style.RESET_ALL)
                    continue
                end_time = time.time() - start_time
                print(f'Training Time: {end_time:.2f} seconds')

                # 验证
                if it % config.train.val_freq == 0:
                    transition_epoch = None
                    avg_scaffold_loss, avg_rgroup_loss = validate(it, model, val_loader, logger, writer,
                                                                  scaffold_epochs, transition_epoch, total_epoch,
                                                                  device)
                    if avg_scaffold_loss is None or avg_rgroup_loss is None:
                        print(f"Fold {fold + 1} Epoch {it} Validation Failed, Skipping")
                        continue

                    # 更新调度器
                    scheduler_scaffold, scheduler_rgroup = schedulers
                    scheduler_scaffold.step(avg_scaffold_loss)
                    scheduler_rgroup.step(avg_rgroup_loss)

                    current_lr_scaffold = optimizers[0].param_groups[0]['lr']
                    logger.info(f"Fold {fold + 1} Epoch {it}, Scaffold LR: {current_lr_scaffold}")
                    writer.add_scalar(f'learning_rate/fold_{fold + 1}/scaffold', current_lr_scaffold, it)

                    current_lr_rgroup = optimizers[1].param_groups[0]['lr']
                    logger.info(f"Fold {fold + 1} Epoch {it}, Rgroup LR: {current_lr_rgroup}")
                    writer.add_scalar(f'learning_rate/fold_{fold + 1}/rgroup', current_lr_rgroup, it)

                    # 计算总损失
                    avg_val_loss = avg_scaffold_loss + avg_rgroup_loss

                    # 先记录旧的最优
                    old_best_scaffold = best_scaffold_loss
                    old_best_rgroup = best_rgroup_loss

                    # 先判断有没有进步
                    scaffold_improved = (avg_scaffold_loss < old_best_scaffold)
                    rgroup_improved = (avg_rgroup_loss < old_best_rgroup)

                    # 分别更新各自最优
                    if scaffold_improved:
                        best_scaffold_loss = avg_scaffold_loss

                    if rgroup_improved:
                        best_rgroup_loss = avg_rgroup_loss

                    if scaffold_improved:
                        best_scaffold_loss = avg_scaffold_loss
                        scaffold_ckpt_path = os.path.join(ckpt_dir, f'fold_{fold + 1}_best_scaffold_{it}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizers': [opt.state_dict() for opt in optimizers],
                            'schedulers': [sch.state_dict() for sch in schedulers],
                            'iteration': it,
                            'avg_scaffold_loss': avg_scaffold_loss,
                            'avg_rgroup_loss': avg_rgroup_loss,
                            'avg_val_loss': avg_val_loss,
                            'random_seed': seed,  # 保存随机种子
                        }, scaffold_ckpt_path)
                        print(f'Saved Best Scaffold Model to {scaffold_ckpt_path}! '
                              f'Scaffold Loss: {avg_scaffold_loss:.4f}')

                    # 若俩都比各自历史最优好
                    if scaffold_improved and rgroup_improved:
                        best_joint_loss = avg_val_loss
                        best_ckpt_path = os.path.join(ckpt_dir, f'fold_{fold + 1}_best_joint_{it}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizers': [opt.state_dict() for opt in optimizers],
                            'schedulers': [sch.state_dict() for sch in schedulers],
                            'iteration': it,
                            'avg_scaffold_loss': avg_scaffold_loss,
                            'avg_rgroup_loss': avg_rgroup_loss,
                            'avg_val_loss': avg_val_loss,
                            'random_seed': seed,  # 保存随机种子
                        }, best_ckpt_path)
                        print(f'Saved Best Joint Model to {best_ckpt_path}! '
                              f'Scaffold Loss: {avg_scaffold_loss:.4f}, Rgroup Loss: {avg_rgroup_loss:.4f}')

                print(Fore.YELLOW + f"========== Fold {fold + 1} Epoch {it} Ended ==========" + Style.RESET_ALL)

            except Exception as e:
                logger.error(f"Fold {fold + 1} Epoch {it} Exception: {e}")
                raise e
