import sys
from datetime import datetime

import math
import torch
import torch.nn.functional as F
import numpy as np
from model import const

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def log(*args):
    print(f'[{datetime.now()}]', *args)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask):
    """
    Subtract center of mass of scaffolds from coordinates of all atoms
    """
    if center_of_mass_mask.dim() == 2:  # 检查是否为 [batch_size, num_atoms]
        center_of_mass_mask = center_of_mass_mask.unsqueeze(-1)
    x_masked = x * center_of_mass_mask
    N = center_of_mass_mask.sum(1, keepdims=True)
    mean = torch.sum(x_masked, dim=1, keepdim=True) / N
    if node_mask.dim() == 2:
        node_mask = node_mask.unsqueeze(-1)
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    if center_of_mass_mask.dim() == 2:  # 检查是否为 [batch_size, num_atoms]
        center_of_mass_mask = center_of_mass_mask.unsqueeze(-1)
    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Partial mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    # TODO: check it
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    if node_mask.dim() == 2:
        node_mask = node_mask.unsqueeze(-1)
    x_masked = x * node_mask
    return x_masked


def concatenate_features(x, h):
    xh = torch.cat([x, h['categorical']], dim=2)
    if 'integer' in h:
        xh = torch.cat([xh, h['integer']], dim=2)
    return xh


def split_features(z, n_dims, num_classes, include_charges):
    assert z.size(2) == n_dims + num_classes + include_charges
    x = z[:, :, 0:n_dims]
    h = {'categorical': z[:, :, n_dims:n_dims+num_classes]}
    if include_charges:
        h['integer'] = z[:, :, n_dims+num_classes:n_dims+num_classes+1]

    return x, h


# For gradient clipping

class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')
    return grad_norm


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


class FoundNaNException(Exception):
    def __init__(self, x, h):
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)

        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        idx = set()
        for i in range(z.shape[0]):
            if torch.any(torch.isnan(z[i])):
                idx.add(i)
        return idx


def get_batch_idx_for_animation(batch_size, batch_idx):
    batch_indices = []
    mol_indices = []
    for idx in [0]:
        if idx // batch_size == batch_idx:
            batch_indices.append(idx % batch_size)
            mol_indices.append(idx)
    return batch_indices, mol_indices


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()
def get_bond_lengths(positions, mask):
    """计算相邻原子的键长"""
    # 占位符：需基于 edge_mask 或邻接矩阵实现
    return torch.zeros_like(positions[..., 0]).unsqueeze(-1)

def get_bond_angles(positions, mask):
    """计算连续三个原子形成的键角"""
    # 占位符：需使用向量运算实现
    return torch.zeros_like(positions[..., 0]).unsqueeze(-1)



def calc_bond_lengths(x, edge_index):
    """
    计算所有成键原子对的键长
    参数:
        x: 原子坐标 (batch_size, num_nodes, 3)
        edge_index: 键的邻接矩阵 (batch, 2, num_edges)
    返回:
        bond_lengths: 键长张量 (batch_size, num_edges)
    """
    batch_size = x.size(0)

    # 提取边的起点和终点索引，形状均为 (batch, num_edges)
    rows = edge_index[:, 0, :]
    cols = edge_index[:, 1, :]

    # 利用 batch 索引来获取每个 batch 中对应的原子坐标
    # 构造一个 batch 索引张量，其形状为 (batch, 1)，方便广播到 (batch, num_edges)
    batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)

    # 分别索引起始原子和终止原子的坐标，结果形状为 (batch, num_edges, 3)
    x_rows = x[batch_indices, rows]
    x_cols = x[batch_indices, cols]

    # 计算两点之间的差值，再计算欧氏距离
    diff = x_rows - x_cols  # (batch, num_edges, 3)
    bond_lengths = torch.norm(diff, dim=-1)  # (batch, num_edges)

    return bond_lengths


def get_bond_length_range(self, atoms, bond_type):
    """
    根据原子类型和键类型返回键长范围。

    参数:
        atoms: 元组 (atom_type1, atom_type2)
        bond_type: 键类型 (1=单键, 2=双键, 3=三键)

    返回:
        min_d, max_d: 键长范围 (Å)
    """
    # 示例键长表（可扩展）
    bond_lengths = {
        ('C', 'C'): {1: (1.50, 1.58), 2: (1.32, 1.38), 3: (1.18, 1.22)},
        ('C', 'O'): {1: (1.40, 1.46), 2: (1.20, 1.24)},
        ('C', 'N'): {1: (1.45, 1.50), 2: (1.28, 1.32)}
    }
    key = tuple(sorted([const.IDX2ATOM[atoms[0].item()], const.IDX2ATOM[atoms[1].item()]]))
    ranges = bond_lengths.get(key, {}).get(bond_type, (1.0, 2.0))  # 默认范围
    return ranges[0], ranges[1]


def reconstruct_from_bond_lengths(x, target_lengths, edge_index, lr=0.1, steps=3):
    """
    通过迭代调整原子位置逼近目标键长
    参数:
        x: 原始坐标 (batch_size, num_nodes, 3)
        target_lengths: 目标键长 (batch_size, num_edges)
        edge_index: 键邻接矩阵 (batch_size, num_nodes, num_nodes, 1)
        lr: 学习率
        steps: 调整步数
    返回:
        x_optimized: 优化后的坐标
    """
    x_optimized = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_optimized], lr=lr)

    for _ in range(steps):
        current_lengths = calc_bond_lengths(x_optimized, edge_index)
        loss = F.mse_loss(current_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return x_optimized.detach()


def calc_bond_angles(x, triplets):
    """
    计算连续三个原子形成的键角
    参数:
        x: 原子坐标 (batch_size, num_nodes, 3)
        triplets: 原子三元组索引 (num_angles, 3) [A,B,C]
    返回:
        angles: 键角张量 (batch_size, num_angles)
    """
    # 提取三个原子坐标
    A = x[:, triplets[:, 0]]  # (batch_size, num_angles, 3)
    B = x[:, triplets[:, 1]]
    C = x[:, triplets[:, 2]]

    # 计算向量
    BA = A - B
    BC = C - B

    # 计算夹角（弧度转角度）
    cosine_angle = torch.sum(BA * BC, dim=-1) / (
            torch.norm(BA, dim=-1) * torch.norm(BC, dim=-1) + 1e-8
    )
    angles = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0)) * 180 / math.pi

    return angles  # (batch_size, num_angles)


def reconstruct_from_geometry(x, target_lengths, target_angles,
                              edge_index, triplets,
                              bond_lr=0.1, angle_lr=0.05, steps=5):
    """
    联合优化键长和键角
    参数:
        x: 原始坐标
        target_lengths: 目标键长 (batch_size, num_edges)
        target_angles: 目标键角 (batch_size, num_angles)
        edge_index: 键邻接矩阵
        triplets: 原子三元组索引
    """
    x_optimized = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_optimized], lr=bond_lr)

    for _ in range(steps):
        # 计算当前几何参数
        current_lengths = calc_bond_lengths(x_optimized, edge_index)
        current_angles = calc_bond_angles(x_optimized, triplets)

        # 计算联合损失
        bond_loss = F.mse_loss(current_lengths, target_lengths)
        angle_loss = F.mse_loss(current_angles, target_angles)
        total_loss = bond_loss + 0.8 * angle_loss  # 权重可调

        # 梯度更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return x_optimized.detach()

def center_pos_pl(ligand_pos, pocket_pos, ligand_batch, pocket_batch):
    ligand_pos_center = ligand_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[ligand_batch]
    pocket_pos_center = pocket_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[pocket_batch]
    return ligand_pos_center, pocket_pos_center

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, coalesce
from torch_geometric.nn import radius_graph, radius

def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    扩展图的高阶邻居连接。
    Args:
        num_nodes: 节点数量（如原子数）。
        edge_index: 原始边索引，形状 (2, E)。
        edge_type: 原始边类型，形状 (E,)。
        order: 高阶扩展的阶数。
    Returns:
        new_edge_index: 扩展后的边索引。
        new_edge_type: 扩展后的边类型。
    """
    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
        return order_mat

    # 假设原始边类型最大值为 5，高阶边从 6 开始
    num_types = 6  # 修改为我的任务中边类型的总数
    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)
    # 高阶边的类型从 num_types 开始递增
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()  # 确保原始边和高阶边不冲突
    type_new = type_mat + type_highorder
    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)

    return new_edge_index, new_edge_type

def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, is_sidechain=None):
    """
    根据空间距离扩展图的边，动态赋值边类型。
    Args:
        pos: 节点坐标，形状 (N, 3)。
        edge_index: 原始边索引，形状 (2, E)。
        edge_type: 原始边类型，形状 (E,)。
        cutoff: 距离阈值。
        batch: 批处理向量，形状 (N,)，0 表示配体，1 表示口袋。
        is_sidechain: 可选，区分侧链节点。
    Returns:
        new_edge_index: 扩展后的边索引。
        new_edge_type: 扩展后的边类型。
    """
    assert edge_type.dim() == 1
    N = pos.size(0)
    bgraph_adj = torch.sparse.LongTensor(
        edge_index,
        edge_type,
        torch.Size([N, N])
    )

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)
        # 根据 batch 判断边的来源
        edge_source = batch[rgraph_edge_index[0]]
        edge_target = batch[rgraph_edge_index[1]]
        # 边类型定义：
        # - 配体-配体：6
        # - 口袋-口袋：7
        # - 配体-口袋 或 口袋-配体：8
        new_edge_type = torch.where(edge_source == edge_target,
                                    edge_source + 6,  # 同类节点（配体-配体 或 口袋-口袋）
                                    torch.full_like(edge_source, 8))  # 异类节点
    else:
        # 处理侧链的逻辑保持不变
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]
        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]
        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1)
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]
        # 为侧链相关边赋值类型（假设为 9）
        new_edge_type = torch.full((rgraph_edge_index.size(1),), 9, device=pos.device)

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        new_edge_type.long(),
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type

def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch, order=3, cutoff=10.0,
                              extend_order=True, extend_radius=True, is_sidechain=None, pocket=False):
    """
    综合高阶和空间扩展图。
    Args:
        同上。
    Returns:
        edge_index: 最终边索引。
        edge_type: 最终边类型。
    """
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            order=order
        )

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            is_sidechain=is_sidechain
        )

    return edge_index, edge_type
