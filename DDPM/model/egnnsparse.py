import time
from typing import List, Optional

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from attention import BasicTransformerBlock
from egnn_pytorch import SiLU, fourier_encode_dist, CoorsNorm, exists, embedd_token

# 全局线性注意力类（保持不变）
import torch
import torch.nn as nn
from einops import rearrange
import torch_geometric
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3)
        self.to_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x, context, mask=None, frag_mask=None):
        h, d = self.heads, self.dim_head

        # to_qkv(x) 将输入特征映射到查询、键和值上，[batch_size, num_nodes, heads * dim_head * 3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 然后通过 chunk(3, dim=-1) 将其分为三个部分：q（查询）、k（键）和 v（值）。这些部分的形状分别是 [batch_size, num_nodes, heads * dim_head]。 rearrange 将其重塑为多头的形式：
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        context_qkv = self.to_qkv(context).chunk(3, dim=-1)
        context_q, context_k, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), context_qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, context_k)

        # 增强 frag 节点的注意力权重
        if frag_mask is not None:
            frag_mask = frag_mask.unsqueeze(1).unsqueeze(2)  # [b, 1, 1, n_context]
            dots = dots + frag_mask.float() * 1.0  # 增加 frag 节点的注意力分数

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [b, 1, 1, n_context]
            dots = dots.masked_fill(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, context_v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_Sparse(Attention):
    def sparse_forward(self, x, context, batch_x=None, batch_context=None, mask=None, frag_mask=None):
        assert batch_x is not None and batch_context is not None, "必须提供 batch_x 和 batch_context 参数"
        batch_uniques, counts = torch.unique(batch_x, return_counts=True)
        batch_size = len(batch_uniques)

        if batch_size == 1:
            x = x.unsqueeze(0)
            context = context[batch_context == batch_uniques[0]].unsqueeze(0)
            if mask is not None:
                mask = mask[batch_context == batch_uniques[0]].unsqueeze(0)
            if frag_mask is not None:
                frag_mask = frag_mask[batch_context == batch_uniques[0]].unsqueeze(0)
            out = self.forward(x, context, mask=mask, frag_mask=frag_mask)
            return out.squeeze(0)
        else:
            x_list = []
            for bi in batch_uniques:
                x_batch = x[batch_x == bi].unsqueeze(0)  # [1, n_batch_x, dim]
                context_batch = context[batch_context == bi].unsqueeze(0)  # [1, n_batch_context, dim]
                mask_batch = mask[batch_context == bi].unsqueeze(0) if mask is not None else None
                frag_mask_batch = frag_mask[batch_context == bi].unsqueeze(0) if frag_mask is not None else None
                out_batch = self.forward(x_batch, context_batch, mask=mask_batch, frag_mask=frag_mask_batch)
                x_list.append(out_batch.squeeze(0))
            return torch.cat(x_list, dim=0)
class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)
        self.attn2 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, mask=None, frag_mask=None):
        """
        全局线性注意力层，支持批次处理和掩码
        :param x: 节点特征 [N, dim]
        :param queries: 全局标记 [num_tokens, dim]
        :param batch: 批次索引 [N]
        :param mask: 掩码 [N]，True 表示有效节点
        :param frag_mask: frag 掩码 [N]，增强 frag 节点的引导
        :return: 更新后的节点特征和全局标记 (x, queries)
        """
        res_x, res_queries = x, queries
        x = self.norm_seq(x)
        queries = self.norm_queries(queries)

        batch_size = len(torch.unique(batch)) if batch is not None else 1
        num_tokens = queries.size(0)

        queries_expanded = queries.unsqueeze(0).expand(batch_size, num_tokens, -1)
        queries_flat = queries_expanded.reshape(batch_size * num_tokens, -1)
        token_batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_tokens)


        # 第一层注意力：全局标记查询节点特征
        induced = self.attn1.sparse_forward(
            queries_flat, x, batch_x=token_batch,batch_context=batch, mask=mask, frag_mask=frag_mask
        )


        mask_induced = torch.ones(induced.size(0), dtype=torch.bool, device=induced.device)  # [8]
        frag_mask_induced = torch.zeros(induced.size(0), dtype=torch.float32, device=induced.device)  # [8]
        # 第二层注意力：节点特征查询 induced
        out = self.attn2.sparse_forward(
            x, induced, batch_x=batch, batch_context=token_batch,mask=mask_induced, frag_mask=frag_mask_induced
        )

        x = out + res_x
        queries = induced.view(batch_size, num_tokens, -1).mean(dim=0) + res_queries

        x_norm = self.ff_norm(x)
        x = self.ff(x_norm) + x_norm
        return x, queries

# EGNN_Sparse 类（保持不变，略）
class EGNN_Sparse(MessagePassing):
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
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.cutoff = cutoff

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )
        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None
        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj = None, edge_attr: OptTensor = None, batch: Adj = None,
                rem_mask: Adj = None, angle_data: List = None, size: Size = None, linker_mask=None,pocket_mask:Adj=None,frag_mask:Adj=None) -> Tensor:
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        if edge_index is None:
            edge_index = radius_graph(coors, self.cutoff, batch=batch, loop=False)
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')
        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors, batch=batch,
                                              rem_mask=rem_mask, pocket_mask=pocket_mask,linker_mask=linker_mask,frag_mask=frag_mask)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message_for_coors(self, x_i, x_j, edge_attr, edge_index, pocket_mask=None, frag_mask=None) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        # 增强pocket和frag的引导
        if pocket_mask is not None:
            pocket_mask_j = pocket_mask[edge_index[1]]
            m_ij = m_ij * (1.0 + pocket_mask_j.float().unsqueeze(-1) * 2.0)
        if frag_mask is not None:
            frag_mask_j = frag_mask[edge_index[1]]
            m_ij = m_ij * (1.0 + frag_mask_j.float().unsqueeze(-1) * 2.0)
        return m_ij

    def message(self, x_i, x_j, edge_attr,edge_index,pocket_mask=None) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)

        if pocket_mask is not None:
            pocket_mask_j = pocket_mask[edge_index[1]]  # [num_edges]
            m_ij = m_ij * (1.0 + pocket_mask_j.float().unsqueeze(-1) * 2.0)

        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # m_ij = self.message(**msg_kwargs)
        if self.update_coors:
        #   更新方向由 rel_coors 提供，幅度由 coor_wij 控制。
            x_i = kwargs["x"][edge_index[0]]
            x_j = kwargs["x"][edge_index[1]]
            edge_attr = kwargs["edge_attr"]
            pocket_mask = kwargs.get("pocket_mask", None)
            frag_mask = kwargs.get("frag_mask", None)
            m_ij_coors = self.message_for_coors(
                x_i=x_i, x_j=x_j, edge_attr=edge_attr, edge_index=edge_index, pocket_mask=pocket_mask,frag_mask=frag_mask
            )
            coor_wij = self.coors_mlp(m_ij_coors)

            if self.coor_weights_clamp_value:
                coor_wij.clamp_(min=-self.coor_weights_clamp_value, max=self.coor_weights_clamp_value)
        # 归一化:将 rel_coors 的每个向量除以其模（加上一个小的 scale 以避免除零）确保相对坐标的长度为 1（或接近 1），只保留方向信息
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])
       # 每条边的相对坐标 rel_coors[i] 乘以标量权重 coor_wij[i]，得到加权方向向量。根据edge_index 将边消息聚合到目标节点，将每条边的加权相对坐标聚合到目标节点，生成每个节点的坐标更新量。
            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            rem_mask = kwargs.get('rem_mask', None)
            if rem_mask is not None:
                # 只更新 rem 部分的坐标
                coors_out = kwargs["coors"] + mhat_i * rem_mask.unsqueeze(1).float()
            else:
                coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        if self.update_feats:
            m_ij_feats = self.message(**msg_kwargs)
            m_i = self.aggregate(m_ij_feats, **aggr_kwargs)
            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            if rem_mask is not None:
                hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
                # 仅在 rem_mask 为 1 的地方更新特征
                hidden_out = kwargs["x"] + hidden_out * rem_mask.unsqueeze(1).float()
            else:
                hidden_out = kwargs["x"] + self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        else:
            hidden_out = kwargs["x"]

        return self.update((hidden_out, coors_out), **update_kwargs)

# 修改后的 EGNN_Sparse_Network 类
class EGNN_Sparse_Network(nn.Module):
    def __init__(
            self,
            n_layers: int,
            feats_input_dim: int,
            feats_dim: int,
            pos_dim: int = 3,
            edge_attr_dim: int = 0,
            m_dim: int = 16,
            fourier_features: int = 0,
            soft_edge: int = 0,
            embedding_nums: List[int] = [],
            embedding_dims: List[int] = [],
            edge_embedding_nums: List[int] = [],
            edge_embedding_dims: List[int] = [],
            update_coors: bool = True,
            update_feats: bool = True,
            norm_feats: bool = True,
            norm_coors: bool = False,
            norm_coors_scale_init: float = 1e-2,
            dropout: float = 0.0,
            coor_weights_clamp_value: Optional[float] = None,
            aggr: str = "add",
            global_linear_attn_every: int = 1,
            global_linear_attn_heads: int = 8,
            global_linear_attn_dim_head: int = 64,
            num_global_tokens: int = 4,
            recalc: int = 0,
            context_dim: Optional[int] = None,  # 新增：context 的维度
            condition_time: bool = True,
            time_dim=128,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.feats_input_dim = feats_input_dim
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim

        # 节点特征嵌入层
        self.emb = nn.Linear(feats_input_dim, feats_dim)

        self.output_map = nn.Linear(feats_dim, feats_input_dim)  # 128 -> 10
        self.emb_norm = nn.LayerNorm(feats_dim)


        self.condition_time = condition_time
        self.time_dim = time_dim
        if self.condition_time:
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



        # 嵌入层（若有）
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i], embedding_dim=embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()
        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i], embedding_dim=edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1

        # 多层 EGNN_Sparse
        self.mpnn_layers = nn.ModuleList()
        for i in range(n_layers):
            layer = EGNN_Sparse(
                feats_dim=feats_dim,
                pos_dim=pos_dim,
                edge_attr_dim=edge_attr_dim,
                m_dim=m_dim,
                fourier_features=fourier_features,
                soft_edge=soft_edge,
                norm_feats=norm_feats,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                update_feats=update_feats,
                update_coors=update_coors,
                dropout=dropout,
                cutoff=10.0,
                aggr=aggr
            )
            is_global_layer = global_linear_attn_every > 0 and (i % global_linear_attn_every == 0)
            if is_global_layer:
                attn_layer = GlobalLinearAttention_Sparse(
                    dim=feats_dim,
                    heads=global_linear_attn_heads,
                    dim_head=global_linear_attn_dim_head
                )
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            else:
                self.mpnn_layers.append(layer)

        # 全局注意力参数
        self.has_global_attn = global_linear_attn_every > 0
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, feats_dim))    # 4,feats_dim的学习参数

    def get_sinusoidal_embedding(self, t, dim):
        """生成正弦位置编码"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # [N, dim]

    def forward(
            self,
            h: Tensor,  # [batch, num_atoms, feats_input_dim]
            pos: Tensor,  # [batch, max_num_atoms, pos_dim]
            context: Optional[Tensor] = None,  # [batch, num_atoms, context_dim]
            batch_mask: Optional[Tensor] = None,  # [batch*num_atoms]
            node_mask: Optional[Tensor] = None,  # [batch, num_atoms, 1]
            rem_mask: Optional[Tensor] = None , # [batch, num_atoms, 1]
            frag_mask: Optional[Tensor] = None ,
            pocket_mask: Optional[Tensor] = None,
            t:torch.Tensor = None
    ):
        # 获取批次大小和原子数
        batch_mask = batch_mask.long()


        batch_size, num_atoms = h.size(0), h.size(1)
        N_total = batch_size * num_atoms


        # 展平输入数据
        h_flat = h.view(N_total, -1)  # [batch*num_atoms, feats_input_dim]
        pos_flat = pos.view(N_total, -1)  # [batch*num_atoms, pos_dim]


        # 处理 batch_mask
        if batch_mask is None:
            batch_mask = torch.arange(batch_size, device=h.device).repeat_interleave(num_atoms)

        # 处理 node_mask 和 rem_mask
        if node_mask is None:
            node_mask_flat = torch.ones(N_total, device=h.device).bool()
        else:
            node_mask_flat = node_mask.view(-1).bool()

        # 筛选有效原子
        valid_indices = torch.where(node_mask_flat)[0]
        h_valid = h_flat[valid_indices]  # [N_valid, feats_input_dim]
        pos_valid = pos_flat[valid_indices]  # [N_valid, pos_dim]
        batch_valid = batch_mask[valid_indices]  # [N_valid]
        # 假设 rem_mask 是 [batch, num_atoms, 1] 的张量

        rem_mask_flat = rem_mask.view(-1).bool()  # 展平为 [batch*num_atoms]
        rem_valid_mask = rem_mask_flat[valid_indices]  # 筛选有效原子，形状 [N_valid]



        frag_mask_flat = frag_mask.view(-1).bool()
        frag_valid_mask = frag_mask_flat[valid_indices]


        pocket_mask_flat = pocket_mask.view(-1).bool()
        pocket_valid_mask = pocket_mask_flat[valid_indices]


        h_valid = self.emb(h_valid)  # [N_valid, feats_dim]
        h_valid = self.emb_norm(h_valid)
        if self.embedding_dims:
            h_valid = embedd_token(h_valid, self.embedding_dims, self.emb_layers)


        if self.condition_time and t is not None:
            # 验证 t 的形状
            assert t.dim() == 2 and t.size(1) == 1, f"Expected t shape [batch_size, 1], got {t.shape}"
            assert t.size(0) == batch_size, f"t batch size ({t.size(0)}) must match h batch size ({batch_size})"

            if t.numel() == 1:
                # 标量时间步：所有有效节点使用相同的值
                t_valid = torch.full((valid_indices.size(0),), t.item(), device=h.device, dtype=h.dtype)
            else:
                # 批次相关时间步：每个样本的原子共享相同的 t 值
                # 扩展 t 到每个样本的所有原子
                t_expanded = t.expand(batch_size, num_atoms).contiguous().reshape(N_total)
                # 筛选有效节点的时间值
                t_valid = t_expanded[valid_indices]  # [N_valid]

            # 正弦时间嵌入
            t_embed = self.get_sinusoidal_embedding(t_valid, self.time_dim)  # [N_valid, time_dim]
            h_time = self.time_embed(t_embed)  # [N_valid, feats_dim]

            if rem_valid_mask is not None:
                h_time = h_time * rem_valid_mask.view(-1, 1).float()

            h_with_time = torch.cat([h_valid, h_time], dim=-1)  # [N_valid, feats_dim*2]
            h_valid = self.fusion_mlp(h_with_time)  # [N_valid, feats_dim


        # 构建图
        edge_index = radius_graph(pos_valid, r=10.0, batch=batch_valid, loop=False)
        edge_attr = None
        if self.edge_embedding_dims:
            edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)

        # 拼接坐标和特征
        x = torch.cat([pos_valid, h_valid], dim=-1)  # [N_valid, pos_dim + feats_dim]


        # 多层消息传递
        for i, layer in enumerate(self.mpnn_layers):
            if isinstance(layer, nn.ModuleList):
                egnn_layer, attn_layer = layer
                x = egnn_layer(x, edge_index, edge_attr, batch=batch_valid, rem_mask=rem_valid_mask,pocket_mask=pocket_valid_mask)
                pos_updated = x[:, :self.pos_dim]
                feats = x[:, self.pos_dim:]
                # 仅使用原始的 global_tokens，不扩展到批次维度

                feats, updated_tokens = attn_layer(feats, self.global_tokens, batch=batch_valid, mask=rem_valid_mask,frag_mask=frag_valid_mask)
                x = torch.cat([pos_updated, feats], dim=-1)
            else:
                x = layer(x, edge_index, edge_attr, batch=batch_valid, rem_mask=rem_valid_mask,pocket_mask=pocket_valid_mask,frag_mask=frag_valid_mask)

            # TODO: 如果需要 recalc 边特征，可以在这里添加逻辑

        # 分离更新后的坐标和特征
        pos_updated = x[:, :self.pos_dim]  # [N_valid, pos_dim]
        h_updated = x[:, self.pos_dim:]  # [N_valid, feats_dim]
        h_updated = self.emb_norm(h_updated)

        h_updated_10d = self.output_map(h_updated)  # [N_valid, output_dim]

        h_valid_10d = self.output_map(h_valid)  # [N_valid, output_dim]


        pos_diff = pos_updated - pos_valid
        h_diff = h_updated_10d - h_valid_10d

        # 提取 rem 部分的更新结果
        rem_valid_indices = torch.where(rem_mask_flat[valid_indices])[0]
        h_diff_rem = h_diff[rem_valid_indices]
        pos_diff_rem = pos_diff[rem_valid_indices]

        return h_diff_rem, pos_diff_rem

    def __repr__(self):
        return f'EGNN_Sparse_Network of {len(self.mpnn_layers)} layers'

# 数据准备
def prepare_data():
    batch_size = 2
    max_num_atoms = 8
    feats_input_dim = 4
    pos_dim = 3
    context_dim = 2

    h = torch.tensor([
        # 批次 0: [frag, rem, rgroup, pocket, pad]
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],  # frag
         [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0],  # rem
         [0.0, 0.0, 0.0, 0.0],  # rgroup
         [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0],  # pocket
         [0.0, 0.0, 0.0, 0.0]],  # pad
        # 批次 1
        [[25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0],  # frag
         [33.0, 34.0, 35.0, 36.0], [37.0, 38.0, 39.0, 40.0],  # rem
         [0.0, 0.0, 0.0, 0.0],  # rgroup
         [41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0],  # pocket
         [0.0, 0.0, 0.0, 0.0]]  # pad
    ], dtype=torch.float32)

    pos = torch.tensor([
        # 批次 0
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # frag
         [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],  # rem
         [0.0, 0.0, 0.0],  # rgroup
         [4.0, 0.0, 0.0], [5.0, 0.0, 0.0],  # pocket
         [0.0, 0.0, 0.0]],  # pad
        # 批次 1
        [[6.0, 0.0, 0.0], [7.0, 0.0, 0.0],  # frag
         [8.0, 0.0, 0.0], [9.0, 0.0, 0.0],  # rem
         [0.0, 0.0, 0.0],  # rgroup
         [10.0, 0.0, 0.0], [11.0, 0.0, 0.0],  # pocket
         [0.0, 0.0, 0.0]]  # pad
    ], dtype=torch.float32)

    context = torch.tensor([
        # 批次 0
        [[0.1, 0.2], [0.3, 0.4],  # frag
         [0.5, 0.6], [0.7, 0.8],  # rem
         [0.0, 0.0],  # rgroup
         [0.9, 1.0], [1.1, 1.2],  # pocket
         [0.0, 0.0]],  # pad
        # 批次 1
        [[1.3, 1.4], [1.5, 1.6],  # frag
         [1.7, 1.8], [1.9, 2.0],  # rem
         [0.0, 0.0],  # rgroup
         [2.1, 2.2], [2.3, 2.4],  # pocket
         [0.0, 0.0]]  # pad
    ], dtype=torch.float32)

    batch_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    node_mask = torch.tensor([
        [[1], [1], [1], [1], [1], [1], [1], [0]],  # 批次 0
        [[1], [1], [1], [1], [1], [1], [1], [0]]   # 批次 1
    ], dtype=torch.float32)

    pocket_mask = torch.tensor( [
        [[0], [0], [0], [0], [0], [1], [1], [0]],  # 批次 0: rem 在索引 2, 3
        [[0], [0], [0], [0], [0], [1], [1], [0]]
    ], dtype=torch.float32)  # 批次 1: rem 在索引 2, 3)
    rem_mask = torch.tensor([
        [[0], [0], [1], [1], [0], [0], [0], [0]],  # 批次 0: rem 在索引 2, 3
        [[0], [0], [1], [1], [0], [0], [0], [0]]   # 批次 1: rem 在索引 2, 3
    ], dtype=torch.float32)
    frag_mask = torch.tensor( [
        [[1], [1], [0], [0], [0], [0], [0], [0]],  # 批次 0: rem 在索引 2, 3
        [[1], [1], [0], [0], [0], [0], [0], [0]] ], dtype=torch.float32)
    return h, pos, context, batch_mask, node_mask, rem_mask,pocket_mask,frag_mask

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EGNN_Sparse_Network(
        n_layers=2,
        feats_input_dim=4,
        feats_dim=8,
        pos_dim=3,
        m_dim=16,
        norm_feats=True,
        norm_coors=True,
        dropout=0.1,
        global_linear_attn_every=1,
        context_dim=2
    ).to(device)

    h, pos, context, batch_mask, node_mask, rem_mask,pocket_mask,frag_mask = prepare_data()
    h = h.to(device)
    pos = pos.to(device)
    context = context.to(device)
    batch_mask = batch_mask.to(device)
    node_mask = node_mask.to(device)
    rem_mask = rem_mask.to(device)
    pocket_mask = pocket_mask.to(device)
    frag_mask = frag_mask.to(device)

    # 记录原始坐标以验证更新
    pos_original = pos.clone()

    try:
        h_diff_rem, pos_diff_rem = model(h, pos, context, batch_mask, node_mask, rem_mask,pocket_mask,frag_mask)
        print("Feature diff for rem part:", h_diff_rem.shape)  # 预计 [4, 8]
        print("Position diff for rem part:", pos_diff_rem.shape)  # 预计 [4, 3]
        print("Feature diff values:", h_diff_rem)
        print("Position diff values:", pos_diff_rem)

        # 验证坐标更新
        pos_flat = pos.view(-1, 3)
        node_mask_flat = node_mask.view(-1).bool()
        valid_indices = torch.where(node_mask_flat)[0]
        pos_valid = pos_flat[valid_indices]
        pos_updated = pos_valid + torch.zeros_like(pos_valid)  # 模拟更新后的坐标
        pos_updated[rem_mask.view(-1).bool()[valid_indices]] += pos_diff_rem

        print("\nVerification:")
        print("Original rem coordinates (batch 0, indices 2,3):", pos[0, 2:4])
        print("Original rem coordinates (batch 1, indices 2,3):", pos[1, 2:4])
        print("Updated rem coordinates:", pos_updated[rem_mask.view(-1).bool()[valid_indices]])
        print("Non-rem coordinates (should be unchanged):")
        print("Frag (batch 0):", pos[0, 0:2])
        print("Frag (batch 1):", pos[1, 0:2])
        print("Pocket (batch 0):", pos[0, 5:7])
        print("Pocket (batch 1):", pos[1, 5:7])
    except Exception as e:
        print(f"Error occurred: {e}")

