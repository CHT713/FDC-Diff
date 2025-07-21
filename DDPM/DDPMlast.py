import copy
import logging
import time
import random

import math
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_
from model import visualizer as vis
from model import function
from torch import nn
import torch.nn.functional as F
from model.egnn1 import DynamicsWithPockets, Dynamics
from model.egnn import DynamicsWithPockets_rgroup
from edm import EDM
from datasets import *
from typing import Dict, List


def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")


class TrainingConfig:
    lambda_kl = 0.0  # KL散度权重
    lambda_t = 0.0  # 时间相关损失权重
    lambda_0 = 0.0  # 初始状态损失权重
    lambda_l2 = 1.0  # L2基础损失权重


class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
            self,
            in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
            inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
            diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
            normalize_factors, include_charges, model,
            data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs,
            n_stability_samples,
            normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
            center_of_mass='scaffold', inpainting=False, anchors_context=True,
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = float(lr)  #
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        self.is_geom = True

        self.stage_params = TrainingConfig()
        self.lambda_kl = self.stage_params.lambda_kl
        self.lambda_t = self.stage_params.lambda_t
        self.lambda_0 = self.stage_params.lambda_0
        self.lambda_l2 = self.stage_params.lambda_l2

        # context_node_nf_rgroup = context_node_nf -1
        context_node_nf_rgroup = context_node_nf - 1

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics_scaffold = DynamicsWithPockets_rgroup(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,  # 从传入的dynamics中获取参数
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )

        dynamics_rgroup = DynamicsWithPockets_rgroup(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf_rgroup,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )
        self.edm = EDM(
            dynamics_scaffold=dynamics_scaffold,  # 替换原来的 dynamics
            dynamics_rgroup=dynamics_rgroup,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,  # #  [1,4,10]
        )

    # def configure_optimizers(self):
    #     base_lr = 2e-4
    #     min_lr = 1e-6  # η_min
    #     T0 = 800  # 第一次重启周期（epoch）
    #     betas = (0.95 ,0.999)
    #
    #     # -------- optimizers --------
    #     opt_scaf = torch.optim.AdamW(
    #         self.edm.dynamics_scaffold.parameters(),
    #         lr=base_lr, weight_decay=1e-4, betas=betas
    #
    #     )
    #     opt_rgrp = torch.optim.AdamW(
    #         self.edm.dynamics_rgroup.parameters(),
    #         lr=base_lr, weight_decay=1e-4, betas=betas
    #     )
    #
    #     # -------- schedulers --------
    #     sch_scaf = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         opt_scaf, T_0=T0, T_mult=2, eta_min=min_lr
    #     )
    #     sch_rgrp = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         opt_rgrp, T_0=T0, T_mult=2, eta_min=min_lr
    #     )
    #
    #     return (
    #         [opt_scaf, opt_rgrp],
    #         [
    #             {"scheduler": sch_scaf, "interval": "epoch", "name": "lr_scaffold"},
    #             {"scheduler": sch_rgrp, "interval": "epoch", "name": "lr_rgroup"},
    #         ]
    #     )

    def configure_optimizers(self):
        optimizer_scaffold = torch.optim.AdamW(
            self.edm.dynamics_scaffold.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        optimizer_rgroup = torch.optim.AdamW(
            self.edm.dynamics_rgroup.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        return [optimizer_scaffold, optimizer_rgroup], []

    def _preprocess_coordinates(self, x, h, node_mask, attachment, fragment_mask,
                                scaffold_mask, rgroup_mask, pocket_mask, training, current_epoch, scaffold_epochs,
                                id=None):
        """根据阶段预处理坐标，返回 x 和 context"""

        com_mask = fragment_mask
        """根据阶段预处理坐标，返回 x 和 context"""
        if id == 0:
            # Scaffold 阶段
            x = x * (1 - rgroup_mask)

            # 将 R 基团的特征设置为 0
            h = h * (1 - rgroup_mask)

            node_mask = node_mask * (1 - rgroup_mask)
            frag_attachment = torch.logical_and(attachment.bool(), fragment_mask.bool()).float()
            # context=fragment_mask+pocket_mask
            context = torch.cat([frag_attachment, fragment_mask, pocket_mask], dim=-1)
        else:
            # anchor_mask =  torch.logical_and(attachment.bool(), scaffold_mask.bool()).float()

            context = torch.cat([scaffold_mask, pocket_mask], dim=-1)

        if com_mask.sum() > 1e-3:
            x = function.remove_partial_mean_with_mask(x, node_mask, com_mask)
            function.assert_partial_mean_zero_with_mask(x, node_mask, com_mask)
        if training and self.data_augmentation:
            x = function.random_rotation(x)

        return x, h, context

    def forward(self, data, training=True, current_epoch=None, scaffold_epochs=None, total_epoch=None):
        try:

            node_mask = data['atom_mask']
            edge_mask = data['edge_mask']
            fragment_mask = data['fragment_mask']
            remaining_mask = data['rem_mask']
            rgroup_mask = data['rgroup_mask']
            scaffold_mask = data['scaffold_mask']
            pocket_mask = data['pocket_mask']
            attachment = data['attachment']
            edge_index = data['scaf_edge_index']

            # 始终计算骨架部分的损失
            x_scaffold, h_scaffold, context_scaffold = self._preprocess_coordinates(
                data['positions'], data['one_hot'], node_mask, attachment, fragment_mask, scaffold_mask, rgroup_mask,
                pocket_mask, training, current_epoch, scaffold_epochs, id=0
            )
            node_mask_scaffold = node_mask * (1 - rgroup_mask)

            out_scaffold = self.edm.forward(
                x=x_scaffold, h=h_scaffold, node_mask=node_mask_scaffold, fragment_mask=fragment_mask,
                remaining_mask=remaining_mask, rgroup_mask=rgroup_mask, scaffold_mask=scaffold_mask,
                edge_mask=edge_mask,
                context=context_scaffold, task_id=0, pocket_mask=pocket_mask,
            )

            x_rgroup, h_rgroup, context_rgroup = self._preprocess_coordinates(
                data['positions'], data['one_hot'], node_mask, attachment, fragment_mask, scaffold_mask, rgroup_mask,
                pocket_mask, training, current_epoch, scaffold_epochs, id=1
            )

            out_rgroup = self.edm.forward(
                x=x_rgroup, h=h_rgroup, node_mask=node_mask, fragment_mask=None, remaining_mask=None,
                scaffold_mask=scaffold_mask, rgroup_mask=rgroup_mask, pocket_mask=pocket_mask,
                edge_mask=edge_mask, context=context_rgroup, task_id=1
            )
            return out_scaffold[4], out_rgroup[4]

        except Exception as e:
            print(f"前向传播错误，Epoch {current_epoch}: {str(e)}")
            raise

    def sample_chain(self, data, sample_fn=None, keep_frames=None, id=None, generated_scaffold=None):
        with torch.no_grad():
            if id == 0:

                # 生成骨架
                print("进入最新版本的 sample_chain")
                self.edm.dynamics_rgroup.eval()  # 冻结 R 基团预测器
                if sample_fn is None:
                    rem_sizes = data['rem_mask'].sum(1).view(-1).int()
                else:
                    rem_sizes = sample_fn(data)

                template_data = create_templates_for_rgroup_generation_single(data, rem_sizes, id)

                x = template_data['positions']

                node_mask = template_data['atom_mask']

                rgroup_mask = template_data['rgroup_mask']

                node_mask = node_mask * (1 - rgroup_mask)

                node_mask_ret = template_data['atom_mask'] - template_data['pocket_mask']
                edge_mask = template_data['edge_mask']
                h = template_data['one_hot']
                rem_mask = template_data['rem_mask']
                attachment = template_data['attachment']
                fragment_mask = template_data['fragment_mask']
                pocket_mask = template_data['pocket_mask']
                num_atoms = template_data['num_atoms']

                fragment_anchors_mask = torch.logical_and(attachment.bool(), fragment_mask.bool()).float()
                context = torch.cat([fragment_anchors_mask, fragment_mask, pocket_mask], dim=-1)
                # context = fragment_mask + pocket_mask

                com_mask = fragment_mask
                x = function.remove_partial_mean_with_mask(x, node_mask, com_mask)

                chain = self.edm.sample_scaf_chain(
                    x=x, h=h, node_mask=node_mask, edge_mask=edge_mask, fragment_mask=fragment_mask,
                    rem_mask=rem_mask, context=context, keep_frames=keep_frames, pocket_mask=pocket_mask, id=id,
                )


                return chain, node_mask_ret, num_atoms, template_data['atom_mask'], template_data['pocket_mask'], \
                template_data['rgroup_mask'], template_data
            else:
                # 生成 R 基团
                print("进入下一阶段")

                self.edm.dynamics_scaffold.eval()  # 冻结骨架预测器
                if sample_fn is None:
                    rgroup_sizes = data['rgroup_mask'].sum(1).view(-1).int()
                else:
                    rgroup_sizes = sample_fn(data)

                if generated_scaffold is not None:

                    x_scaffold = generated_scaffold['positions']
                    h_scaffold = generated_scaffold['one_hot']
                    # 更新 data 中对应骨架区域的位置信息和特征
                    # 这里需要根据具体的mask（例如data中可能有fragment_mask和rem_mask）
                    # 假设data['positions']的前几部分对应frag_pos和rem_pos
                    # 可以用以下伪代码进行替换：
                    new_positions = data['positions'].clone()
                    new_one_hot = data['one_hot'].clone()
                    # 根据mask确定需要替换的索引，这里以scaffold_mask为例
                    scaffold_idx = data['scaffold_mask'].bool()

                    # num_scaffold_nodes = scaffold_idx.sum().item()  # 计算 True 的数量
                    # assert num_scaffold_nodes == x_scaffold.size(1), f"scaffold 节点数量不匹配: {num_scaffold_nodes} vs {x_scaffold.size(1)}"

                    scaffold_indices = torch.where(scaffold_idx[0])[0]
                    new_positions[0, scaffold_indices, :] = x_scaffold[0, scaffold_indices, :]
                    new_one_hot[0, scaffold_indices, :] = h_scaffold[0, scaffold_indices, :]

                    # scaffold_indices = torch.where(scaffold_idx.squeeze(0).squeeze(-1))[0]
                    # new_positions[0, scaffold_indices, :] = x_scaffold[0]  # 假设 batch 维度为 0
                    # new_one_hot[0, scaffold_indices, :] = h_scaffold[0]
                    # 用修改后的数据继续构造模板
                    data_modified = data.copy()

                    data_modified['positions'] = new_positions
                    data_modified['one_hot'] = new_one_hot

                    template_data = create_templates_for_rgroup_generation_single(data_modified, rgroup_sizes, id)



                else:
                    template_data = create_templates_for_rgroup_generation_single(data, rgroup_sizes, id)
                    print("错误执行")

                # 如果需要更新其他字段（如掩码），在这里添加
                x = template_data['positions']

                node_mask = template_data['atom_mask']

                node_mask_ret = template_data['atom_mask'] - template_data['pocket_mask']

                edge_mask = template_data['edge_mask']
                h = template_data['one_hot']
                scaffold_mask = template_data['scaffold_mask']
                pocket_mask = template_data['pocket_mask']
                rgroup_mask = template_data['rgroup_mask']
                num_atoms = template_data['num_atoms']

                anchors = template_data['fragment_mask']



                context = torch.cat([scaffold_mask, pocket_mask], dim=-1)

                x = function.remove_partial_mean_with_mask(x, node_mask, anchors)



                chain = self.edm.sample_chain(
                    x=x, h=h, node_mask=node_mask, edge_mask=edge_mask, scaffold_mask=scaffold_mask,
                    rgroup_mask=rgroup_mask, context=context, keep_frames=keep_frames, pocket_mask=pocket_mask, id=id
                )
                return chain, node_mask_ret, num_atoms