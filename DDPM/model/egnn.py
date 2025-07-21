import logging
import time

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import function


def setup_logging(log_file='egnn_log1.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO ,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # 确保日志文件在每次运行时清空（可选）
    open(log_file, 'w').close()


setup_logging()

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


class MLPEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim=100, activation='relu'):    #   hidden_dim: 128  mlp_act: relu
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim // 1, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type=None):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, hidden_dim)
        """
        d_emb = self.mlp(edge_length)  # (num_edge, hidden_dim)
        if edge_type is None:
            return d_emb
        edge_attr = self.bond_emb(edge_type)  # (num_edge, hidden_dim)
        return d_emb * edge_attr  # (num_edge, hidden_dim)


class GCL(nn.Module):
    # 这段代码实现了一个图卷积层（Graph Convolution Layer, GCL），用于处理图结构数据。它基于 PyTorch 框架，通过定义一个 GCL 类来更新图中节点的特征表示
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2   # 256
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf), # 输入（256+2，输出128）
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        elif normalization == 'layer_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.LayerNorm(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.LayerNorm(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())
 # 边特征计算：通过 edge_model 函数，将源节点和目标节点的特征（以及可选的边属性）结合起来，经过一个多层感知机（MLP）生成边特征
    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)  # 拼接之后的维度为[num_edges,128+128+2】
        # 实际上处理的是将【源节点特征（source），目标节点特征（target），边属性（edge_attr）】进行拼接输入到mlp中进行处理
        # 为什么要进行拼接？综合考虑边的上下文信息：边的特征不仅仅依赖于边本身的属性（edge_attr），还需要结合源节点和目标节点的特征来捕捉边的语义。
        # 例如，在一个社交网络中，边的特征（比如"朋友关系强度"）可能依赖于两个用户的属性（如年龄、兴趣）以及边的属性（如交互频率）。
        mij = self.edge_mlp(out)
        # logging.info(f"GCL edge_model mij min/max: {mij.min().item()}/{mij.max().item()}")

      # 它是每条边经过 MLP 处理后的高维表示，融合了源节点、目标节点和边属性的信息
        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij
# 通过 node_model 函数，利用 unsorted_segment_sum 将邻居节点的边特征聚合到每个节点上，然后更新节点特征。
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        # 把边上的消息"汇聚到"节点上的操作
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # logging.info(f"GCL node_model agg min/max: {agg.min().item()}/{agg.max().item()}")

        # edge_feat[0]（边 0->1）加到节点 0。edge_feat[1]（边 1->2）加到节点 1。
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # agg 现在包含了每个节点的原始特征和从邻居边聚合来的特征，

        out = x + self.node_mlp(agg) # 残差连接，保留了原始特征 x，避免信息丢失，缓解过平滑问题，提高训练稳定性，符合 GNN 的增量更新思想。
        # logging.info(f"GCL node_model out min/max: {out.min().item()}/{out.max().item()}")
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index   #  (2, num_edges)
        # h[row], h[col]表示从edge_index 中取出对应的节点特征。
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    # 用于更新图中节点的坐标（coord），并保持某种等变性
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        # 输出维度设为 1，意味着线性层的输出是一个标量。
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # 这行代码用于初始化 layer.weight 的权重。torch.nn.init.xavier_uniform_ 是一种权重初始化方法，它是根据 Xavier 初始化方法进行均匀分布初始化的。
        # 最后一层使用 Xavier 均匀初始化，gain=0.001，使初始输出较小，避免过大的坐标更新。
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, rgroup_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        # logging.info(f"EquivariantUpdate trans min/max: {trans.min().item()}/{trans.max().item()}")
        if edge_mask is not None:
            trans = trans * edge_mask
        # trans 被聚合到节点上（通过 unsorted_segment_sum），形成每个节点的坐标更新量
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if rgroup_mask is not None:
            agg = agg * rgroup_mask

        coord = coord + agg
        # logging.info(f"EquivariantUpdate coord min/max: {coord.min().item()}/{coord.max().item()}")
        return coord

    def forward(
            self, h, coord, edge_index, coord_diff, edge_attr=None, rgroup_mask=None, node_mask=None, edge_mask=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, rgroup_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    # 这段代码定义了一个名为 EquivariantBlock 的 PyTorch 模块，它结合了图神经网络（GNN）和等变操作，用于同时更新图中节点的特征（h）和坐标（x）
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',task_aware=True):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers    # 2
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant    # 0.000001
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.task_aware = task_aware

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))

        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, rgroup_mask=None, edge_mask=None, edge_attr=None, task_feat=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        # 不使用sin_embedding处理距离，直接使用传入的edge_attr
        edge_attr = torch.cat([distances, edge_attr], dim=1)

        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            rgroup_mask=rgroup_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    # 这段代码定义了一个名为 EGNN（等变图神经网络，Equivariant Graph Neural Network）的 PyTorch 模块，用于同时更新图中节点的特征（h）和坐标（x）。
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', activation=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',task_aware=True):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf    # 14
        self.hidden_nf = hidden_nf       # 128
        self.device = device
        self.n_layers = n_layers               # 6
        self.coords_range_layer = float(coords_range/n_layers)  # 2.5
        self.norm_diff = norm_diff        #TRUE
        self.normalization_factor = normalization_factor    #[1,4,10]
        self.aggregation_method = aggregation_method  # sum
        self.task_aware = task_aware

        if sin_embedding:   # false
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        # # 添加边编码器
        # self.edge_encoder = MLPEdgeEncoder(hidden_dim=hidden_nf, activation='relu')   #  128


        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,task_aware=True))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, rgroup_mask=None, edge_mask=None, task_feat=None):
        # Edit Emiel: Remove velocity as input
        # radial: 边的平方距离 (num_edges, 1)
        distances, _ = coord2diff(x, edge_index)
        # logging.info(f"EGNN distances min/max: {distances.min().item()}/{distances.max().item()}")
        
        # 使用边编码器处理边特征，而不是直接使用距离
        # edge_attr = self.edge_encoder(distances)
        #
        # edge_attr = torch.cat([distances, edge_attr], dim=1)

        h = self.embedding(h)
        # logging.info(f"After embedding h min/max: {h.min().item()}/{h.max().item()}")

        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index,
                node_mask=node_mask,
                rgroup_mask=rgroup_mask,
                edge_mask=edge_mask,
                edge_attr=distances,
                task_feat=task_feat,
            )
            # logging.info(f"After e_block_{i} h min/max: {h.min().item()}/{h.max().item()}")
            # logging.info(f"After e_block_{i} x min/max: {x.min().item()}/{x.max().item()}")
            if torch.isnan(h).any() or torch.isnan(x).any():
                logging.error(f"NaN detected after e_block_{i}")
                break

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        # 每个 batch 处理完后添加分隔线
        logging.info("----------")
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 activation=nn.SiLU(), n_layers=4, attention=False, normalization_factor=1,
                 out_node_nf=None, normalization=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, activation=activation,
                attention=attention, normalization=normalization))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]  # 计算每条边的坐标差
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1) # 沿着维度 1（坐标维度）求和，得到每条边的平方距离，形状
    norm = torch.sqrt(radial + 1e-8)  # 计算平方根，得到每条边的欧几里得距离，形状 (N, 1)。
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class Dynamics(nn.Module):
    def __init__(
            self, n_dims, in_node_nf, context_node_nf, hidden_nf=128, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None, centering=False,task_types=['scaffold', 'rgroup'],task_aware=True,
            time_dim=16,feats_dim=10
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.task_types = task_types  # 新增任务类型参数
        self.task_aware = task_aware  # 任务感知开关

        self.time_dim = time_dim
        self.feats_dim = feats_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, feats_dim),
            nn.SiLU(),
            nn.Linear(feats_dim, feats_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feats_dim * 2, feats_dim),
            nn.SiLU(),
            nn.Linear(feats_dim, feats_dim),
            nn.LayerNorm(feats_dim)  # 融合归一化
        )

        in_node_nf = in_node_nf + context_node_nf + feats_dim
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )


        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        else:
            raise NotImplementedError

        self.edge_cache = {}

    def forward(self, t, xh, node_mask, noise_mask, edge_mask, context,phase,training_stage=None,
              ):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        bs, n_nodes = xh.shape[0], xh.shape[1]
        edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if noise_mask is not None:
            noise_mask = noise_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        # 根据训练阶段选择噪声掩码

        if self.condition_time: # 若 t 是标量，则为所有节点填充相同的时间值。若 t 是向量，则按批量重复并展平。
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1),将时间特征拼接到 h 上。
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)  # 将上下文特征拼接到 h 上。

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':  # 执行这个
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                rgroup_mask=noise_mask,
                edge_mask=edge_mask
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-self.feats_dim]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise function.FoundNaNException(vel, h_final)

        if self.centering:
            vel = function.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)


class DynamicsWithPockets_rgroup(Dynamics):

    def get_sinusoidal_embedding(self, t, dim):
        """
        t: [N] (一维张量)
        return: [N, dim] (sin/cos位置编码)
        """
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        # t: [N], emb: [half_dim] --> broadcasting得到 [N, half_dim]
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # shape: [N, dim]


    def forward(self, t, xh, node_mask, noise_mask, edge_mask, context,task_id):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        # """

        bs, n_nodes = xh.shape[0], xh.shape[1] #xh是节点特征（B,N,D），B是批次大小，N是节点数，D=3
        # 将掩码展平为二维张量，便于后续计算。
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1) 表示将所有数据压缩成一个列向量（每个元素占据一个单独的行）。
        if noise_mask is not None:
           noise_mask = noise_mask.view(bs * n_nodes, 1)  # (B*N, 1)
#        z_t = xh * scaffold_mask.unsqueeze(-1) + z_t * rgroup_mask.unsqueeze(-1)
        # Reshaping node features & adding time feature，.clone() 用于创建 xh 张量的一个副本，确保原始张量 xh 不会被直接修改。

        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)，这里 -1 表示让 PyTorch 自动计算该维度的大小，以确保张量中的元素总数不变。
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        N_total = bs * n_nodes
        # edge_mask 标记每个节点属于哪个batch

        edges = self.get_dist_edges(x, node_mask, edge_mask)  # （2，num_edges)，获取边函数
        if self.condition_time and t is not None:
            # 1) 确保 t.shape = [bs, 1]
            assert t.dim() == 2 and t.size(0) == bs and t.size(1) == 1, \
                f"Expected t shape [bs, 1], got {t.shape}"

            # 2) 将 [bs, 1] 的 t 展开到 [bs*n_nodes]，使每个样本的所有节点共用相同 t 值
            if t.numel() == 1:
                # 全 batch 只有一个时间步，直接重复
                t_expanded = t.repeat(N_total, 1)  # [bs*n_nodes, 1]
            else:
                # 若 batch 内每个样本的 t 不同，则先 expand 到 (bs, n_nodes)，再 flatten
                # t: [bs, 1] -> [bs, n_nodes], 再 reshape -> [bs*n_nodes, 1]
                t_expanded = t.expand(bs,n_nodes).reshape(N_total, 1)

            # 3) 做正弦嵌入：先 squeeze() 让 t_expanded 变成 [N_total] 然后 get_sinusoidal_embedding
            t_embed = self.get_sinusoidal_embedding(
                t_expanded.squeeze(-1),  # -> shape [N_total]
                self.time_dim  # 你定义的时间嵌入维度
            )  # -> shape [N_total, time_dim]

            # 4) (可选) 使用一层 MLP/线性层把 t_embed 再投影到 feats_dim
            # 假设你定义了 self.time_embed = nn.Linear(self.time_dim, some_dim)
            # 或者你想和 h 做 concat，就都行
            t_embed_mapped = self.time_embed(t_embed)  # shape [N_total, some_dim]

            # 5) 直接拼到 h 上
            h = torch.cat([h, t_embed_mapped], dim=-1)  # 现在 h 多了时间的嵌入信息

        # if self.condition_time:
        #     if np.prod(t.size()) == 1:  # 判断 t 是否为单元素；若是，说明所有样本用同一个时间值。
        #         # t is the same for all elements in batch.
        #         h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        #     else:
        #         # t is different over the batch dimension.
        #         h_time = t.view(bs, 1).repeat(1, n_nodes)  # h_time 的形状变为 (bs, n_nodes)。
        #         h_time = h_time.view(bs * n_nodes, 1)
        #     h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf) # 这里指定为3
            h = torch.cat([h, context], dim=1)    # (B*N, nf+1+3)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                rgroup_mask=noise_mask,
                edge_mask=None
                ,)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case 表示坐标的变化量

        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:# 移除上下文特征
            h_final = h_final[:, :-self.context_node_nf]
        # Slice off last dimension which represented time.
        if self.condition_time: # 移除时间特征
            h_final = h_final[:, :-self.feats_dim]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            # raise function.FoundNaNException(vel, h_final)
            logging.warning("NaN values detected in vel or h_final. Replacing with zeros.")
            vel = torch.nan_to_num(vel, nan=0.0)
            h_final = torch.nan_to_num(h_final, nan=0.0)

        if self.centering:
            vel = function.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    @staticmethod
    def get_dist_edges(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges
