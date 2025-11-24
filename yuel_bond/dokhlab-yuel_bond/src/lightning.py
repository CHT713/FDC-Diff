import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb

from src import utils, const
from src.gnn import GNN
from src.const import N_RESIDUE_TYPES, TORCH_INT
from src.datasets import (
    BondDataset, get_dataloader, collate
)
from src.utils import sum_except_batch
from typing import Dict, List, Optional
from tqdm import tqdm
from torch.nn import functional as F
from pdb import set_trace
import torch.nn as nn

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

class YuelBond(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        data_path, train_data_prefix, val_data_prefix, hidden_nf, has_bonds=False,
        activation='silu', tanh=False, n_layers=3, attention=False, norm_constant=1,
        inv_sublayers=2, sin_embedding=True, normalization_factor=1, aggregation_method='sum',
        normalize_factors=True, model=None,
        batch_size=2, lr=1e-4, torch_device='cpu', test_epochs=1, n_stability_samples=1,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='fragments', anchors_context=True, graph_type=None, 
    ):
        super(YuelBond, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.has_bonds = has_bonds
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation

        self.in_node_nf = len(const.ALLOWED_ATOM_TYPES)
        self.in_edge_nf = 1
        self.hidden_nf = hidden_nf
        self.out_node_nf = 0
        self.out_edge_nf = len(const.RDKIT_BOND_TYPES)

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        if type(activation) is str:
            activation = get_activation(activation)

        # in_node_nf, in_edge_nf, hidden_nf, 
        self.gnn = GNN(
            # n_layers=3, attention=False,
                 # norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=hidden_nf,
            out_node_nf=self.out_node_nf,
            out_edge_nf=self.out_edge_nf,
            n_layers=n_layers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
        )

    def setup(self, stage: Optional[str] = None):
        dataset_type = BondDataset
        if stage == 'fit':
            self.train_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device,
                has_bonds=self.has_bonds,
            )
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device,
                has_bonds=self.has_bonds,
            )
        elif stage == 'val':
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device,
                has_bonds=self.has_bonds,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, collate_fn=collate):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def forward(self, data):
        h = data['one_hot']
        edge_index = data['edge_index'].to(TORCH_INT)
        edge_attr = data['edge_attr']

        node_mask = data['node_mask']
        edge_mask = data['edge_mask']

        # feat_mask = torch.tensor(molecule_feat_mask(), device=x.device)

        _, edge_pred = self.gnn.forward(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,

            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        return edge_pred
    
    def loss_fn(self, edge_pred, edge_true, edge_mask):
        # edge_pred的最后一维中选最大的那个位置作为预测的bond_order，
        # 然后计算预测的bond_order和真实的bond_order的交叉熵
        # 如果edge_mask为0，则不计算loss，因为有可能这个位置没有edge，只是padding
        # edge_pred: b, n, c
        # edge_true: b, n, c
        # edge_mask: b, n, 1
        batch_size, n_bonds, n_bond_types = edge_pred.shape
        # 这里用 view 是因为 cross entropy 要求输入的维度是 (batch_size * n_bonds, n_bond_types)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
        edge_pred = edge_pred.view(batch_size * n_bonds, n_bond_types)
        edge_true = edge_true.view(batch_size * n_bonds, n_bond_types)
        edge_mask = edge_mask.view(batch_size * n_bonds)
        # 这里用 long 是因为 cross entropy 要求 target 是 long 类型
        edge_true = edge_true.argmax(dim=-1).long()
        loss = F.cross_entropy(edge_pred, edge_true)
        loss = loss * edge_mask
        loss = loss.sum() / edge_mask.sum()
        return loss

    def training_step(self, data, *args):
        edge_pred = self.forward(data)
        edge_true = data['bond_orders']
        edge_mask = data['edge_mask']

        edge_pred = edge_pred * edge_mask # b, n, c

        loss = self.loss_fn(edge_pred, edge_true, edge_mask)

        training_metrics = {
            'loss': loss,
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)

        # self.training_step_outputs.append(training_metrics)
        return training_metrics

    def validation_step(self, data, *args):
        edge_pred = self.forward(data)
        edge_true = data['bond_orders']
        edge_mask = data['edge_mask']

        edge_pred = edge_pred * edge_mask # b, n, c

        loss = self.loss_fn(edge_pred, edge_true, edge_mask)

        rt = {
            'loss': loss,
        }
        self.validation_step_outputs.append(rt)
        return rt

    def test_step(self, data, *args):
        edge_pred = self.forward(data)
        edge_true = data['bond_orders']
        edge_mask = data['edge_mask']

        edge_pred = edge_pred * edge_mask # b, n, c

        loss = self.loss_fn(edge_pred, edge_true, edge_mask)

        rt = {
            'loss': loss,
        }
        self.test_step_outputs.append(rt)
        return rt

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
