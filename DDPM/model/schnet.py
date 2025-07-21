from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph

from .attention import BasicTransformerBlock
from common import GaussianSmearing, ShiftedSoftplus

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0, smooth=False):
        super().__init__(aggr='add') # 调用父类 MessagePassing 的初始化方法，并设置消息聚合方法为加法（'add'）。这意味着在进行消息传递时，会将所有接收到的消息相加。
        self.lin1 = Linear(in_channels, num_filters, bias=False) # 定义一个线性层 lin1，用于将输入特征从 in_channels 映射到 num_filters，不使用偏置（bias=False）。
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )  # Network for generating filter weights
        self.cutoff = cutoff    # cutoff: 用于截断边特征的距离阈值。
        self.smooth = smooth    # smooth: 一个布尔值，指示是否使用平滑处理

# forward函数在神经网络中用于定义前向传播的计算过程，它接收输入数据并通过网络的各层进行处理，最终输出预测结果。
    def forward(self, x, edge_index, edge_length, edge_attr):

        W = self.nn(edge_attr)    # 通过定义的序列网络 nn 处理边特征 edge_attr，生成边的权重 W

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)   # 如果使用平滑，计算平滑权重 C，使用余弦函数将边长度归一化到 [0, 1] 范围。
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff  # 将 C 限制在截止距离 cutoff 以内，超出范围的权重设为 0
        else:
            C = (edge_length <= self.cutoff).float()   # 如果不使用平滑，直接根据边长度判断是否在截止距离以内，生成一个布尔张量，并转换为浮点数（在截止内为 1，外部为 0）。
        # if self.cutoff is not None:
        #     C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
        #     C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        W = W * C.view(-1, 1)   # 将生成的权重 W 乘以平滑权重 C。C.view(-1, 1) 使得 C 的形状与 W 兼容，以便进行逐元素相乘。

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)   # 调用 propagate 方法进行消息传递，将处理后的特征 x 和权重 W 传递给与之相连的节点。
        x = self.lin2(x)
        return x    # 代表经过图卷积和特征变换后的节点特征

    def message(self, x_j, W):
        return x_j * W     # 返回加权的邻居节点特征，即将邻居特征 x_j 乘以对应的权重 W


class InteractionBlock(Module):      # 用于构建交互模块，通常用于处理图中的节点交互。

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth=False):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff, smooth)
        # 实例化一个 CFConv 对象，使用隐藏通道数、滤波器数量等参数来构建卷积层。
        self.act = ShiftedSoftplus()   # 定义激活函数为 ShiftedSoftplus，用于增加非线性。
        self.lin = Linear(hidden_channels, hidden_channels)
        # self.bn = BatchNorm(hidden_channels)
        # self.gn = GraphNorm(hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        # x = self.bn(x)
        # x = self.gn(x)
        return x


class SchNetEncoder_protein(Module):
# 该模型处理原子属性、原子位置以及它们之间的边特征（例如，原子间的距离），并通过多次“交互”过程来更新原子特征表示。
# 该网络使用了高斯扩展和基于图的消息传递机制，能够有效地捕捉原子之间的距离和其他边特征对分子表示的影响。
    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=64, cutoff=10.0, input_dim=27):   # 128,128,3,    ,10, false , 10 ,false ,true
        super().__init__()

        self.hidden_channels = hidden_channels  # 128  隐藏层通道数
        self.num_filters = num_filters    # 128        滤波器数量
        self.num_interactions = num_interactions    # 3  交互层
        self.input_dim = input_dim    # 10

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        # 实例化一个高斯扩展对象，用于根据边的长度生成边特征。使用 GaussianSmearing 类生成边特征。这个类将根据原子之间的距离计算高斯扩展，用于捕捉原子之间的相对距离信息。
        self.cutoff = cutoff   # 10


        self.emblin = Linear(self.input_dim, hidden_channels) # 一个线性变换层，将输入的原子属性（node_attr）从 input_dim 映射到 hidden_channels 维度的隐藏表示。10 ---128
        # interactions是一个包含多个 InteractionBlock 的列表，每个 InteractionBlock 执行一次消息传递操作。每次交互都会更新节点的特征表示。
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth=True)
            self.interactions.append(block)

    @property
    def out_channels(self):    # 定义一个只读属性 out_channels，返回隐藏通道数
        return self.hidden_channels
#
    def forward(self, node_attr, pos, batch):
        # radius_graph 根据节点之间的欧几里得距离，连接那些距离小于某个给定半径 (radius) 的节点。它返回一个图的边列表，边的连接是基于这些节点之间的距离是否小于设定的半径。
        # torch_geometric.nn.radius_graph(x, r, batch=None, loop=False, max_num_neighbors=None)
        # x (Tensor): 输入节点特征，形状为 (N, F)，其中 N 是节点数，F 是特征维度。 batch (LongTensor, 可选): 每个节点所属图的批次索引，如果有多个图，必须提供此参数。
        # loop (bool, 可选): 是否允许节点和自己连接。如果设置为 True，则每个节点都会有一个自连接边。返回一个边列表，通常是一个包含边的索引的长整型张量（edge_index）。这个张量的形状为 (2, E)，其中 E 是边的数量。每一列表示一条边，边的两个端点由两行表示。
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = self.emblin(node_attr)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # batch = batch.squeeze(0)
        # # print(batch.size())
        # # print(h.size())
        # h = scatter_mean(h, batch, dim=0)
        # h = h.index_select(0, batch_ligand)
        return h


class SchNetEncoder(Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5, time_emb=True,
                 context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.emblin = Linear(self.input_dim, hidden_channels)  # 16 or 8
        self.context = context

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
        if context:
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                z, ptemb = z[:, :self.input_dim], z[:, self.input_dim:]
                h = self.emblin(z) + ptemb
            else:
                h = self.emblin(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h


class CASchNetEncoder(Module):
    '''
    cross attention schnet encoder
    '''

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5,
                 n_head=8, d_dim=32, time_emb=True, context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.emblin = Linear(self.input_dim, hidden_channels)  # 16 or 8
        self.context = context

        self.interactions = ModuleList()
        self.crossattns = ModuleList()
        self.atten_layer = BasicTransformerBlock(hidden_channels, n_head, d_dim, 0.1, hidden_channels)
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
            # atten_layer = BasicTransformerBlock(hidden_channels,n_head,d_dim,0.1,hidden_channels)
            # self.crossattns.append(atten_layer)

        if context:
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, p_ctx, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                z, ptemb = z[:, :self.input_dim], z[:, self.input_dim:]
                h = self.emblin(z) + ptemb
            else:
                h = self.emblin(z)
        # h = self.atten_layer(h,p_ctx)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # for interaction,crossattn in zip(self.interactions,self.crossattns):
        #     h = crossattn(h,p_ctx)
        #     h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h


class SchNetEncoder_pure(Module):
    '''
    cross attention schnet encoder
    '''

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5,
                 time_emb=True, context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.context = context

        self.interactions = ModuleList()

        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)

        if context:
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, p_ctx, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            h = z  # default
            # if self.time_emb:
            #     z, ptemb = z[:,:self.input_dim],z[:,self.input_dim:]
            #     h = self.emblin(z)+ptemb
            # else:
            #     h = self.emblin(z)
        # h = self.atten_layer(h,p_ctx)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # for interaction,crossattn in zip(self.interactions,self.crossattns):
        #     h = crossattn(h,p_ctx)
        #     h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h
import torch

