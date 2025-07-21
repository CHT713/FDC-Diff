# coding=utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch_geometric.nn import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj, remove_self_loops
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_sparse import coalesce




# from torch_scatter import scatter_mean, scatter_add


def split_tensor_by_batch(x, batch, num_graphs=None):
    """
    Args:
        x:      (N, ...)
        batch:  (B, )
    Returns:
        [(N_1, ), (N_2, ) ..., (N_B, ))]
    """
    if num_graphs is None:
        num_graphs = batch.max().item() + 1
    x_split = []
    for i in range(num_graphs):
        mask = batch == i
        x_split.append(x[mask])
    return x_split


def concat_tensors_to_batch(x_split):
    x = torch.cat(x_split, dim=0)
    batch = torch.repeat_interleave(
        torch.arange(len(x_split)),
        repeats=torch.LongTensor([s.size(0) for s in x_split])
    ).to(device=x.device)
    return x, batch


def split_tensor_to_segments(x, segsize):
    num_segs = math.ceil(x.size(0) / segsize)
    segs = []
    for i in range(num_segs):
        segs.append(x[i * segsize: (i + 1) * segsize])
    return segs


def split_tensor_by_lengths(x, lengths):
    segs = []
    for l in lengths:
        segs.append(x[:l])
        x = x[l:]
    return segs


def batch_intersection_mask(batch, batch_filter):
    batch_filter = batch_filter.unique()
    mask = (batch.view(-1, 1) == batch_filter.view(1, -1)).any(dim=1)
    return mask


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, input, batch, num_graphs):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, batch, dim=0, dim_size=num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, input, batch, num_graphs):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, batch, dim=0, dim_size=num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_protein = torch.cat([
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx


def get_complete_graph(batch):
    """
    Args:
        batch:  Batch index.
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        neighbors:  (B, ), number of edges per graph.
    """
    natoms = scatter_add(torch.ones_like(batch), index=batch, dim=0)

    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms  # Number of edges per graph

    return edge_index, num_edges


def compose_context_stable(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = batch_ligand.max().item() + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    for i in range(num_graphs):
        mask_p, mask_l = (batch_protein == i), (batch_ligand == i)
        batch_p, batch_l = batch_protein[mask_p], batch_ligand[mask_l]

        batch_ctx += [batch_p, batch_l]
        h_ctx += [h_protein[mask_p], h_ligand[mask_l]]
        pos_ctx += [pos_protein[mask_p], pos_ligand[mask_l]]
        mask_protein += [
            torch.ones([batch_p.size(0)], device=batch_p.device, dtype=torch.bool),
            torch.zeros([batch_l.size(0)], device=batch_l.device, dtype=torch.bool),
        ]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein


# if __name__ == '__main__':
#     h_protein = torch.randn([60, 64])
#     h_ligand = -torch.randn([33, 64])
#     pos_protein = torch.clamp(torch.randn([60, 3]), 0, float('inf'))
#     pos_ligand = torch.clamp(torch.randn([33, 3]), float('-inf'), 0)
#     batch_protein = torch.LongTensor([0]*10 + [1]*20 + [2]*30)
#     batch_ligand = torch.LongTensor([0]*11 + [1]*11 + [2]*11)

#     h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand)

#     assert (batch_ctx[mask_protein] == batch_protein).all()
#     assert (batch_ctx[torch.logical_not(mask_protein)] == batch_ligand).all()

#     assert torch.allclose(h_ctx[torch.logical_not(mask_protein)], h_ligand)
#     assert torch.allclose(h_ctx[mask_protein], h_protein)

#     assert torch.allclose(pos_ctx[torch.logical_not(mask_protein)], pos_ligand)
#     assert torch.allclose(pos_ctx[mask_protein], pos_protein)


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    if edge_attr is not None:
        h_pair = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (E, 2H)
    else:
        h_pair = h_row * h_col
    return h_pair


# def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
#     h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
#     if edge_attr is not None:
#         h_pair = torch.cat([h_row, h_col, edge_attr], dim=-1)    # (E, 3H)
#     else:
#         h_pair = torch.cat([h_row, h_col], dim=-1)   # (E, 2H)
#     return h_pair


def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index, edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)  # (G, )
    node_offset = num_cum_nodes - num_nodes_per_graph  # (G, )
    edge_offset = node_offset[edge2graph]  # (E, )

    num_nodes_square = num_nodes_per_graph ** 2  # (G, )
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (G, )
    edge_start = num_nodes_square_cumsum - num_nodes_square  # (G, )
    edge_start = edge_start[edge2graph]

    all_len = num_nodes_square_cumsum[-1]

    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = node_index.max(dim=-1)[0]
    node_small = node_index.min(dim=-1)[0]
    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

    symm_noise = torch.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (E, 1)
    return d_noise


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        # binarize 函数将输入矩阵中大于 0 的元素变为 1，其它元素变为 0。这是因为在图中，我们只关心是否有连接，而不关心连接的路径数量。

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        #
        #  这个函数用于生成高阶邻接矩阵。它的基本思路是从原始邻接矩阵（表示节点间的直接连接关系）出发，通过矩阵乘法逐步构建出表示高阶邻居连接的矩阵。
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device),   # 单位矩阵，表示每个节点与自己有连接（对角线为 1）。
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        # 原始邻接矩阵加上单位矩阵，然后使用 binarize 函数将矩阵中所有非零元素转为 1，得到1阶邻接矩阵的二进制表示。

        for i in range(2, order + 1):
            # 这里相乘一次实际上是从该点多走一步到另一个点的路径数量，使用binarize将其转为1即可新增一个边来连接
            # 类似，再乘一次，就是多走两步，这里循环两次，也就是走两步，即连接3-hop邻居
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0) # 稀疏的边索引 edge_index 转换为密集的邻接矩阵 adj，并移除多余的维度（如果有的话）
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N) # 得到这个高阶邻接矩阵可以用于进一步分析图中节点的复杂连接关系

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
    # 这行代码的作用是根据输入的边索引 edge_index 和边属性 edge_type 创建一个密集的类型矩阵 type_mat，并移除可能存在的多余维度（如果有的话）。
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    # type_highorder 矩阵标记出了 adj_order 中连接阶数大于 1 的位置，并为其赋予了新的类型编码（6 和 7），而其他位置为 0，表示非高阶连接或不存在连接（根据具体应用场景的定义）。
    assert (type_mat * type_highorder == 0).all() # 其作用是检查 type_mat 和 type_highorder 两个矩阵对应元素相乘的结果是否全部为 0
    type_new = type_mat + type_highorder
    # 其实这里的新矩阵就是原矩阵+新的编码，也就是高阶连接的编码。（就是在原矩阵存在高阶链接的位置标记新的键类型）
    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    # 将密集表示的矩阵 type_new 转换回稀疏表示，得到边索引 new_edge_index 和边类型 new_edge_type。
    _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data

    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added.
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type


# 这里设置unspecified_type_number=-1. 是为了被标记为radius的edge加上原来的edge_type 1 = 0
# 这里设置unspecified_type_number=1，是为了ligand original:2; pocket cutoff:1; ligand cutoff: 3
# protein original: 4; protein cutoff:5

# 这段代码的目的是在现有的图结构上，基于节点的空间位置和一个给定的半径阈值（cutoff），通过添加半径关系来扩展图的边。
def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=1, is_sidechain=None):
    assert edge_type.dim() == 1
    N = pos.size(0)
    # print(edge_index.size())
    # print(edge_type.size())
    # print(N)
    bgraph_adj = torch.sparse.LongTensor(     # 通过原始的 edge_index 和 edge_type 创建一个稀疏矩阵 bgraph_adj，表示原始图的邻接关系。
        edge_index,
        edge_type,
        torch.Size([N, N])
    )

    if is_sidechain is None: # 执行
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)
        # 基于节点的空间位置 pos 和给定的半径 cutoff 来创建图中的边
        # 这个方法返回一个边的索引矩阵，形状为 (2, E_r)，表示新计算出的边。
    else:
        # fetch sidechain and its batch index
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))  # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))  # (2, E)
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1)  # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]
    # 创建一个稀疏张量 rgraph_adj，它表示由 rgraph_edge_index 定义的边的邻接矩阵，并且这些边的类型由 unspecified_type_number 决定。
    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # 将原始图的邻接矩阵 bgraph_adj 与半径图的邻接矩阵 rgraph_adj 相加，并通过 coalesce() 去除重复边（即去掉同一对节点之间的多条边）。coalesce() 的作用是合并稀疏矩阵中的重复项，确保每对节点之间只有一条边。

    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    # 通过 composed_adj 获取更新后的边索引和边类型，作为函数的输出。
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch, order=3, cutoff=10.0,
                              extend_order=True, extend_radius=True, is_sidechain=None, pocket=False):
    if extend_order:    #true
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_type=edge_type, order=order   # 全是1的张量大小为ligand_bond_index.size(1)
        )
        # edge_index_order = edge_index
        # edge_type_order = edge_type
    unspecified_type_number = -1
    if pocket:
        unspecified_type_number = 1

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            unspecified_type_number=unspecified_type_number,
            is_sidechain=is_sidechain

        )

    return edge_index, edge_type


def get_edges(x, batch_mask, ligand_batch, protein_cutoff, ligand_cutoff, ligand_edge=None):
    # TODO: cache batches for each example in self._edges_dict[n_nodes]
    adj_full = batch_mask[:, None] == batch_mask[None, :]  # only consider the edges inside ligand and pocket
    # 通过比较 batch_mask 来生成一个全连接的邻接矩阵 adj_full，其中只有相同批次的节点之间才有连接。

    # adj_full = torch.ones((len(batch_mask),len(batch_mask))).to(x.device)
    # adj_full = adj_full>0

    # construct the corss eges for ligand and pocket
    if protein_cutoff is not None:
        adj = adj_full & (torch.cdist(x, x) <= protein_cutoff)
    # 使用 torch.cdist 计算节点之间的欧氏距离。
    # 将距离小于等于 protein_cutoff 的节点对标记为有边缘连接。 # 得到一个布尔矩阵
    # 结合 adj_full，确保只在同一批次内考虑边缘。  将 adj_full 和上述布尔矩阵按位与（&）操作

    # drop the edges of the pocket protein iteself if we consider cross edges
    # adj[len(ligand_batch):,len(ligand_batch):] = 0

    # construct the inner edges for the ligand
    if ligand_cutoff is not None and ligand_edge is None:
        inner_adj = adj_full & (torch.cdist(x, x) <= ligand_cutoff)   # inner_adj 是一个布尔矩阵，表示在同一批次中且满足距离阈值条件的配体节点对是否有边连接。
        inner_adj = inner_adj[:len(ligand_batch), :len(ligand_batch)]
        # 通过切片操作，提取出配体内部的邻接矩阵：
        # inner_adj[:len(ligand_batch), :len(ligand_batch)]：
        # 提取邻接矩阵的左上角部分，只包含配体节点间的关系。
        # 形状为 (len(ligand_batch), len(ligand_batch))，即配体节点的子矩阵。 只保留配体内部节点之间的边连接，忽略其他节点（如蛋白质节点）
        adj[:len(ligand_batch), :len(ligand_batch)] = inner_adj   # 将配体内部的邻接矩阵 inner_adj 更新到总邻接矩阵 adj 的对应部分。
    else:
        adj[:len(ligand_batch), :len(ligand_batch)] = to_dense_adj(ligand_edge).squeeze(0)

    edges = torch.stack(torch.where(adj), dim=0)
    edges, _ = remove_self_loops(edges)

    return edges


def coarse_grain(pos, node_attr, subgraph_index, batch):
    cluster_pos = scatter_mean(pos, index=subgraph_index, dim=0)  # (num_clusters, 3)
    cluster_attr = scatter_add(node_attr, index=subgraph_index, dim=0)  # (num_clusters, H)
    cluster_batch, _ = scatter_max(batch, index=subgraph_index, dim=0)  # (num_clusters, )

    return cluster_pos, cluster_attr, cluster_batch


def batch_to_natoms(batch):
    return scatter_add(torch.ones_like(batch), index=batch, dim=0)


def get_complete_graph(natoms):
    """
    Args:
        natoms: Number of nodes per graph, (B, 1).
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        num_edges:  (B, ), number of edges per graph.
    """
    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms  # Number of edges per graph

    return edge_index, num_edges
