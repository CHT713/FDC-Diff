from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):     # # 128 , 4 ,32 ,0.1
        super().__init__()
        inner_dim = dim_head * heads   # 定义了多头注意力层的总维度。
        context_dim = default(context_dim, query_dim)
        # 检查 context_dim 是否已指定。
#       使用指定的 context_dim。
  #     如果未指定，将 context_dim 设置为 query_dim。

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

   #  将查询向量 q、键向量 k 和值向量 v 从单头的视角重新排列为多头的形状。
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> (h) n d', h=h), (q, k, v))

         # einsum('h i d,h j d -> h i j', q, k) 是爱因斯坦求和公式，用于高效计算张量间的元素相乘和求和。
        # h i d 表示 q 的形状，    h j d 表示 k 的形状     ，计算结果 h i j 表示头维度下，     ////序列位置 i 与 j 的相似性。
        sim = einsum('h i d,h j d ->h i j', q, k) * self.scale   # 简单的说，这一步就是计算每个头的 查询和所有键之间的点积。
       # mask is None or does not exist
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)


        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('h i j,h j d ->h i d', attn, v)
        out = rearrange(out, '(h) n d -> n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):   # # 128 , 4 ,32 ,0.1 ,128
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,   # 128 , 4 ,32 ,0.1
                                    dropout=dropout)  # is a self-attention for ligand
        # 对配体（ligand）进行自注意力（self-attention）操作，以捕获配体内部的信息。

        self.attn_p = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        # 对蛋白质（protein）的特征进行类似的自注意力操作。

        # 分别为配体和蛋白质设计的前馈网络（FeedForward），用于对特征进行非线性变换。
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.ffp = FeedForward(dim, dropout=dropout, glu=gated_ff)

        # 这两个层用于配体与蛋白质之间的交叉注意力操作，通过互相参考彼此的特征，进一步捕获蛋白质和配体之间的交互关系。
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.attn2p = CrossAttention(query_dim=dim, context_dim=context_dim,
                                     heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        # 分层归一化（LayerNorm）层，分别作用于配体和蛋白质的特征表示，以稳定训练过程和规范化特征。
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm1p = nn.LayerNorm(dim)
        self.norm2p = nn.LayerNorm(dim)
        self.norm3p = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # def forward(self, x, context=None, context_mask=None):
    #     return checkpoint(self._forward, (x, context, context_mask), self.parameters(), self.checkpoint)

    def forward(self, x, context=None, context_mask=None):
        # 这一步应用了 attn1 层，该层是配体特征 x 的自注意力操作。在输入前对 x 进行归一化（self.norm1(x)），然后通过自注意力捕获配体内部的关联。结果加回到 x，形成残差连接，更新后的 x 包含了配体内部的依赖关系。
        x = self.attn1(self.norm1(x)) + x

        # 在配体特征更新后，代码对蛋白质特征 context 进行类似的自注意力操作（attn_p）。这里的 context 代表蛋白质的特征，通过 attn_p 捕获蛋白质内部的关系，更新后的 context 含有蛋白质内部的依赖。
        if context is not None:
            # print('context is not None')
            context = self.attn_p(self.norm1p(context)) + context

        # 在这一步中，代码对 x 应用 attn2 层，这是一个交叉注意力层，用于配体 x 查询蛋白质 context 的特征。这里，配体 x 使用蛋白质 context 作为参考，目的是学习蛋白质对配体的影响，从而捕获两者之间的交互信息。
        if context_mask is not None:
            # print('context_mask is not None')
            x = self.attn2(self.norm2(x), context=context[context_mask, :]) + x
        else:
            x = self.attn2(self.norm2(x), context=context) + x

        # 这一行代码则让蛋白质 context 使用配体 x 作为查询对象，通过 attn2p 捕获配体对蛋白质的影响。这一过程进一步加强了蛋白质对配体交互的理解，并将结果整合回 context，形成蛋白质特征的更新。
        context = self.attn2p(self.norm2p(context), context=x) + context

        # 在交叉注意力完成之后，配体和蛋白质的特征分别通过各自的前馈网络 ff 和 ffp 进行处理。前馈网络对更新后的特征应用非线性变换，进一步丰富特征表示。通过残差连接，保持信息的完整性。
        x = self.ff(self.norm3(x)) + x
        context = self.ffp(self.norm3p(context)) + context
        return x, context

        # return x
