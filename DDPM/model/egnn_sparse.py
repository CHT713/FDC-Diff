from typing import List

from einops import rearrange

from .attention import BasicTransformerBlock

# types


import torch_geometric
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

# except:
#     Tensor = OptTensor = Adj = MessagePassing = Size = object
#     PYG_AVAILABLE = False

#     # to stop throwing errors from type suggestions
#     Adj = object
#     Size = object
#     OptTensor = object
#     Tensor = object

from .egnn_pytorch import *


# global linear attention

class Attention_Sparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None:
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1:
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze()  #  get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi, n_idxs in zip(*batch_uniques):
                x_list.append(
                    self.sparse_forward(
                        x[aux_count:aux_count + n_i],
                        context[aux_count:aux_count + n_idxs],
                        batch_uniques=(bi.unsqueeze(-1), n_idxs.unsqueeze(-1))
                    )
                )
            return torch.cat(x_list, dim=0)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64
    ):
        super().__init__()
        self.norm_seq = torch_geomtric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geomtric.nn.norm.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim, heads, dim_head)
        self.attn2 = Attention_Sparse(dim, heads, dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geomtric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask=mask)
        out = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x = out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries


#  define pytorch-geometric equivalents

class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
            self,
            feats_dim,
            pos_dim=3,
            edge_attr_dim=0,
            m_dim=16,
            fourier_features=0,
            soft_edge=0,
            norm_feats=False,
            norm_coors=False,
            norm_coors_scale_init=1e-2,
            update_feats=True,
            update_coors=True,
            dropout=0.,
            cutoff=10,
            coor_weights_clamp_value=None,
            aggr="add",
            **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None
        self.cutoff = cutoff

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)

        # 根据 dropout 参数构造一个 dropout 层（若丢弃率大于 0），否则使用 Identity 层（即不做变化）。
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #  EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )


#  如果启用了软边（soft_edge），构造一个 MLP 用于根据边消息计算一个标量边权重，并经过 Sigmoid 限制在 (0,1) 之间；否则设置为 None。
        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1),
                                         nn.Sigmoid()
                                         ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.

    #  如果启用节点特征归一化，则构造一个 PyG 版的 LayerNorm 对节点特征进行归一化；否则置为 None。注释中提到不能直接用 Identity，
        #  因为 PyG 的 LayerNorm 需要接收批次信息等额外参数。
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        #  COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj = None,
                edge_attr: OptTensor = None, batch: Adj = None, ligand_batch: Adj = None,
                angle_data: List = None, size: Size = None, linker_mask=None) -> Tensor:
        """ Inputs:
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (n_edges, 2)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        # print(edge_index.size())
        # print(edge_attr.size())
        # coors, feats = coors, h
        if edge_index is None:
            edge_index = radius_graph(coors, self.cutoff, batch=batch, loop=False)
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        # print(rel_dist.size())

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        # print(rel_dist.size())
        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors,
                                               batch=batch, ligand_batch=ligand_batch, linker_mask=linker_mask)
        # output = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
        #                                        coors=coors, rel_coors=rel_coors,
        #                                        batch=batch, ligand_batch=ligand_batch, linker_mask=linker_mask)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)
            # print('*************************')
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        # pyg 2.3.0 compatibility
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                  edge_index, size, kwargs)

        # # pyg version older than 2.3.0
        # size = self.__check_input__(edge_index, size)
        # coll_dict = self.__collect__(self.__user_args__,
        #                              edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        #  get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                # coor_weights.clamp_(min=-clamp_value, max=clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])
            m = coor_wij * kwargs["rel_coors"]
            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            linker_mask = kwargs['linker_mask']
            if linker_mask is not None:
                mhat_i = mhat_i * linker_mask.unsqueeze(1)
            coors_out = torch.cat(
                [kwargs["coors"][:len(kwargs['ligand_batch']), :] + mhat_i[:len(kwargs['ligand_batch']), :],
                 kwargs["coors"][len(kwargs['ligand_batch']):, :]], dim=0)
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            # if self.soft_edge:
            #     m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
            # hidden_out = torch.cat([kwargs["x"][:len(kwargs['ligand_batch']),:] + hidden_out[:len(kwargs['ligand_batch']),:],
            # kwargs["x"][len(kwargs['ligand_batch']):,:]],dim=0)
        else:
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds.
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed.
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed.
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed.
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed.
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """
    def __init__(self, n_layers, feats_input_dim, feats_dim,  # 3,10,128
                 pos_dim=3,
                 edge_attr_dim=0,  # 128
                 m_dim=16,  # 128
                 fourier_features=0,
                 soft_edge=0,  # TRUE
                 embedding_nums=[],
                 embedding_dims=[],
                 edge_embedding_nums=[],
                 edge_embedding_dims=[],
                 update_coors=True,
                 update_feats=True,
                 norm_feats=True,
                 norm_coors=False,  # true
                 norm_coors_scale_init=1e-2,
                 dropout=0.,
                 coor_weights_clamp_value=None,
                 aggr="add",
                 global_linear_attn_every=0,
                 global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64,
                 num_global_tokens=4,
                 recalc=0, ):
        super().__init__()

        self.n_layers = n_layers  # 3

        # Embeddings? solve here
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        # nn.ModuleList() 是一个容器，用于存放子模块（例如层、块等），
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # instantiate point and edge embedding layers
        # 节点和边嵌入：为每个节点和边分别创建嵌入层，允许网络根据节点特征和边特征进行学习。
        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i],
                                                embedding_dim=embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i],
                                                     embedding_dim=edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers = nn.ModuleList()
        self.feats_input_dim = feats_input_dim
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats = update_feats
        self.update_coors = update_coors
        self.dropout = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        # 如果 global_linear_attn_every 参数大于零，则每隔一定层数应用全局注意力机制。该注意力机制可以学习全局节点特征
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))
        # 使用 EGNN_Sparse 层来构建网络。每层的输入和输出都是图的节点特征。每一层都会进行消息传递，更新节点特征。
        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_Sparse(feats_dim=feats_dim,     #128
                                pos_dim=pos_dim,
                                edge_attr_dim=edge_attr_dim,  # 128
                                m_dim=m_dim,    # 128
                                fourier_features=fourier_features,    # 0
                                soft_edge=soft_edge,   # true
                                norm_feats=norm_feats,   # true
                                norm_coors=norm_coors,    # true
                                norm_coors_scale_init=norm_coors_scale_init, # 1e-2
                                update_feats=update_feats,   # true
                                update_coors=update_coors,   # true
                                dropout=dropout,
                                coor_weights_clamp_value=coor_weights_clamp_value)

            # global attention case
            # ： 判断当前层是否需要使用全局注意力层：如果启用了全局注意力且当前层索引能整除 global_linear_attn_every，则认为这是一个全局注意力层。
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
# 如果当前层为全局注意力层，则构造一个全局注意力模块 attn_layer，并将其与当前的 EGNN_Sparse 层组合（放入一个 ModuleList 中一起管理），即当前层同时有消息传递部分和注意力部分。
            if is_global_layer:
                attn_layer = GlobalLinearAttention(dim=self.feats_dim,
                                                   heads=global_linear_attn_heads,
                                                   dim_head=global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            # normal case
            else:
                self.mpnn_layers.append(layer)
        # self.node_mlp = nn.Linear(feats_dim, m_dim)
        self.ligandemb = nn.Linear(feats_input_dim, feats_dim)     # 10---128
        self.proteinemb = nn.Linear(feats_input_dim, feats_dim)
        self.atten_layer = BasicTransformerBlock(feats_dim, 8, feats_dim // 8, 0.1, feats_dim)

    def forward(self, z, pos, edge_index, edge_attr, batch, ligand_batch, context, linker_mask=None,
                bsize=None, recalc_edge=None, verbose=0):  # 前向传播实现了网络的实际计算逻辑。输入数据为节点特征、节点位置、边索引、边特征等信息，输出为更新后的节点特征和位置。
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        h = z
        x = torch.cat([pos, h], dim=1)

        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        #  regulates wether to embedd edges each layer
        edges_need_embedding = False
        # print(len(self.mpnn_layers))
        for i, layer in enumerate(self.mpnn_layers):

            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            #  attn tokens
            global_tokens = None
            # if exists(self.global_tokens):
            #     unique, amounts = torch.unique(batch, return_counts)
            #     num_idxs = torch.cat([torch.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
            #     global_tokens = self.global_tokens[num_idxs]

            #  pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0

            # 若当前层为普通的 EGNN 层，则直接调用该层的 forward 方法，传入当前 x、边索引、边属性以及批次信息。更新 x 后继续传递。
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, ligand_batch=ligand_batch, size=bsize,
                          linker_mask=linker_mask)
            else:

               # 先提取 x 中除坐标之外的特征部分（x[:, self.pos_dim:]），输入到列表中的第一个模块（EGNN 层）和全局 tokens
               # （这里 global_tokens 目前为 None，可能需要进一步设计）。
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                #  merge attn-ed feats and coords
               # 得到经过注意力计算的特征 x_attn 后，再与原始坐标部分拼接恢复成完整输入。
                x = torch.cat((x[:, :self.pos_dim], x_attn), dim=-1)
               # 然后将拼接后的结果传入当前层中的第二个模块（全局注意力层）进行进一步处理。
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            #如果设置了重新计算边特征的参数 (self.recalc 非 0)
            # 在每隔一定层数（i % self.recalc == 0）且不是最后一层时，调用传入的 recalc_edge 函数，利用当前节点 x 重新计算边索引和边属性。
            # 重置标志 edges_need_embedding 为 True，表示下一个层需要对边特征进行嵌入处理。
            if self.recalc and ((i % self.recalc == 0) and not (i == len(self.mpnn_layers) - 1)):
                edge_index, edge_attr, _ = recalc_edge(x)  #  returns attr, idx, any_other_info
                edges_need_embedding = True

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        coors = coors - pos
        # node_embs = self.node_mlp(feats)
        return feats[:len(ligand_batch), :], coors[:len(ligand_batch), :]

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))
