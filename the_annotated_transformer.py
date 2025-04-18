# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% [markdown] id="SX7UC-8jTsp7" tags=[]
#
# <center><h1>The Annotated Transformer</h1> </center>
#
#
# <center>
# <p><a href="https://arxiv.org/abs/1706.03762">Attention is All You Need
# </a></p>
# </center>
#
# <img src="images/aiayn.png" width="70%"/>
#
# * *2022版本: Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak,
#    以及 Stella Biderman.*
# * *[原版](https://nlp.seas.harvard.edu/2018/04/03/attention.html):
#    [Sasha Rush](http://rush-nlp.com/).*
#
#
# 在过去的五年里，Transformer一直是很多人关注的焦点。
# 本文以逐行实现的形式呈现了论文的注解版本。相比原论文，本文重新组织并删除了
# 一些章节，并在整篇文章中添加了注释。这份文档本身就是一个可运行的笔记本，
# 应该是一个完全可用的实现。
# 代码可在[此处](https://github.com/harvardnlp/annotated-transformer/)获取。
#


# %% [markdown] id="RSntDwKhTsp-"
# <h3> 目录 </h3>
# <ul>
# <li><a href="#prelims">准备工作</a></li>
# <li><a href="#background">背景</a></li>
# <li><a href="#part-1-model-architecture">第一部分：模型架构</a></li>
# <li><a href="#model-architecture">模型架构</a><ul>
# <li><a href="#encoder-and-decoder-stacks">编码器和解码器栈</a></li>
# <li><a href="#position-wise-feed-forward-networks">位置前馈网络</a></li>
# <li><a href="#embeddings-and-softmax">嵌入和Softmax</a></li>
# <li><a href="#positional-encoding">位置编码</a></li>
# <li><a href="#full-model">完整模型</a></li>
# <li><a href="#inference">推理</a></li>
# </ul></li>
# <li><a href="#part-2-model-training">第二部分：模型训练</a></li>
# <li><a href="#training">训练</a><ul>
# <li><a href="#batches-and-masking">批处理和掩码</a></li>
# <li><a href="#training-loop">训练循环</a></li>
# <li><a href="#training-data-and-batching">训练数据和批处理</a></li>
# <li><a href="#hardware-and-schedule">硬件和调度</a></li>
# <li><a href="#optimizer">优化器</a></li>
# <li><a href="#regularization">正则化</a></li>
# </ul></li>
# <li><a href="#a-first-example">第一个示例</a><ul>
# <li><a href="#synthetic-data">合成数据</a></li>
# <li><a href="#loss-computation">损失计算</a></li>
# <li><a href="#greedy-decoding">贪婪解码</a></li>
# </ul></li>
# <li><a href="#part-3-a-real-world-example">第三部分：真实世界示例</a>
# <ul>
# <li><a href="#data-loading">数据加载</a></li>
# <li><a href="#iterators">迭代器</a></li>
# <li><a href="#training-the-system">系统训练</a></li>
# </ul></li>
# <li><a href="#additional-components-bpe-search-averaging">附加组件：BPE、搜索、平均</a></li>
# <li><a href="#results">结果</a><ul>
# <li><a href="#attention-visualization">注意力可视化</a></li>
# <li><a href="#encoder-self-attention">编码器自注意力</a></li>
# <li><a href="#decoder-self-attention">解码器自注意力</a></li>
# <li><a href="#decoder-src-attention">解码器源注意力</a></li>
# </ul></li>
# <li><a href="#conclusion">结论</a></li>
# </ul>


# %% [markdown] id="BhmOhn9lTsp8"
# # Prelims
#
# <a href="#background">Skip</a>

# %% id="NwClcbH6Tsp8"
# # !pip install -r requirements.txt

# %% id="NwClcbH6Tsp8"
# # Uncomment for colab
# #
# # !pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
# # !python -m spacy download de_core_news_sm
# # !python -m spacy download en_core_web_sm


# %% id="v1-1MX6oTsp9"
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


# %%
# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# %% [markdown] id="jx49WRyfTsp-"
# > 我的评论以引用块的形式呈现。正文内容均来自论文本身。

# %% [markdown] id="7phVeWghTsp_"
# # 背景

# %% [markdown] id="83ZDS91dTsqA"
#
# 减少序列计算的目标也构成了扩展神经GPU、ByteNet和ConvS2S的基础,这些模型都使用卷积神经网络作为基本构建块,
# 并行计算所有输入和输出位置的隐藏表示。在这些模型中,关联两个任意输入或输出位置的信号所需的操作数随着位置之间的
# 距离而增长,ConvS2S呈线性增长,ByteNet呈对数增长。这使得学习远距离位置之间的依赖关系变得更加困难。在Transformer中,
# 这被减少到恒定数量的操作,尽管代价是由于对注意力加权位置进行平均而导致有效分辨率降低,我们通过多头注意力机制来抵消这种影响。
#
# 自注意力(有时称为内部注意力)是一种将单个序列的不同位置关联起来以计算序列表示的注意力机制。自注意力已经在多种任务中
# 成功应用,包括阅读理解、抽象摘要、文本蕴含和学习与任务无关的句子表示。端到端记忆网络基于循环注意力机制而不是序列对齐的
# 循环机制,已被证明在简单语言问答和语言建模任务上表现良好。
#
# 然而,据我们所知,Transformer是第一个完全依赖自注意力来计算其输入和输出表示的转导模型,不使用序列对齐的RNN或卷积。

# %% [markdown]
# # 第一部分：模型架构

# %% [markdown] id="pFrPajezTsqB"
# # 模型架构

# %% [markdown] id="ReuU_h-fTsqB"
#
# 大多数具有竞争力的神经序列转导模型都具有编码器-解码器结构
# [(引用)](https://arxiv.org/abs/1409.0473)。在这里,编码器将符号表示的输入序列$(x_1, ..., x_n)$映射到
# 连续表示序列$\mathbf{z} = (z_1, ..., z_n)$。给定$\mathbf{z}$,解码器然后一次生成一个符号,
# 生成输出序列$(y_1,...,y_m)$。在每一步,模型都是自回归的
# [(引用)](https://arxiv.org/abs/1308.0850),在生成下一个符号时将先前生成的符号作为额外输入。

# %% id="k0XGXhzRTsqB"
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# %% id="NKGoH2RsTsqC"
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


# %% [markdown] id="mOoEnF_jTsqC"
#
# Transformer遵循这种整体架构，对编码器和解码器都使用堆叠的自注意力层和逐点全连接层，
# 分别如图1的左半部分和右半部分所示。

# %% [markdown] id="oredWloYTsqC"
# ![](images/ModalNet-21.png)


# %% [markdown] id="bh092NZBTsqD"
# ## 编码器和解码器堆叠
#
# ### 编码器
#
# 编码器由$N=6$个相同的层堆叠而成。

# %% id="2gxTApUYTsqD"
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# %% id="xqVTz9MkTsqD"
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# %% [markdown] id="GjAKgjGwTsqD"
#
# 我们在两个子层的周围都使用了残差连接
# [(引用)](https://arxiv.org/abs/1512.03385)，并在其后进行层归一化
# [(引用)](https://arxiv.org/abs/1607.06450)。

# %% id="3jKa_prZTsqE"
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# %% [markdown] id="nXSJ3QYmTsqE"
#
# 也就是说，每个子层的输出是 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，
# 其中 $\mathrm{Sublayer}(x)$ 是由子层本身实现的函数。我们在每个子层的输出上
# 应用 dropout [(引用)](http://jmlr.org/papers/v15/srivastava14a.html)，
# 然后将其与子层的输入相加并进行归一化。
#
# 为了便于实现这些残差连接，模型中的所有子层以及嵌入层都产生维度为
# $d_{\text{model}}=512$ 的输出。

# %% id="U1P7zI0eTsqE"
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# %% [markdown] id="ML6oDlEqTsqE"
#
# 每一层都包含两个子层。第一个是多头自注意力机制，第二个是简单的基于位置的
# 全连接前馈网络。

# %% id="qYkUFr6GTsqE"
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# %% [markdown] id="7ecOQIhkTsqF"
# ### Decoder
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#

# %%
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# %% [markdown] id="dXlCB12pTsqF"
#
# 除了每个编码器层中的两个子层外，解码器还插入了第三个子层，该子层对编码器栈的输出执行多头注意力计算。
# 与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。

# %% id="M2hA1xFQTsqF"
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# %% [markdown] id="FZz5rLl4TsqF"
#
# 我们还修改了解码器栈中的自注意力子层，以防止位置关注后续位置。这种掩码机制，
# 结合输出嵌入偏移一个位置的事实，确保了位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

# %% id="QN98O2l3TsqF"
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


# %% [markdown] id="Vg_f_w-PTsqG"
#
# > 下面的注意力掩码展示了每个目标词（行）被允许关注的位置（列）。
# > 在训练过程中，词语被阻止关注未来的词。

# %% id="ht_FtgYAokC4"
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)

# %% [markdown] id="Qto_yg7BTsqG"
# ### 注意力机制

# 注意力函数可以描述为将一个查询(query)和一组键值对(key-value pairs)映射到一个输出的过程，
# 其中查询、键、值和输出都是向量。输出是值的加权和，其中分配给每个值的权重是通过查询和相应键之间的
# 相容性函数计算得到的。

# 我们将我们特定的注意力机制称为"缩放点积注意力"(Scaled Dot-Product Attention)。
# 输入包含维度为$d_k$的查询和键，以及维度为$d_v$的值。我们计算查询和所有键的点积，
# 将每个点积除以$\sqrt{d_k}$，然后应用softmax函数来获得值的权重。

# ![](images/ModalNet-19.png)


# %% [markdown] id="EYJLWk6cTsqG"
#
# 在实践中，我们同时对一组查询计算注意力函数，将它们打包到一个矩阵$Q$中。键和值也被打包到
# 矩阵$K$和$V$中。我们按如下方式计算输出矩阵：
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$

# %% id="qsoVxS5yTsqG"
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# %% [markdown] id="jUkpwu8kTsqG"
#
# 两种最常用的注意力函数是加性注意力[(引用)](https://arxiv.org/abs/1409.0473)和点积
# (乘性)注意力。点积注意力与我们的算法相同，只是多了一个缩放因子$\frac{1}{\sqrt{d_k}}$。
# 加性注意力使用一个具有单个隐藏层的前馈网络来计算相容性函数。虽然这两种方法在理论复杂度上
# 相似，但点积注意力在实践中更快且空间效率更高，因为它可以使用高度优化的矩阵乘法代码来实现。
#
#
# 虽然对于较小的$d_k$值，这两种机制表现相似，但在没有缩放的情况下，当$d_k$值较大时，
# 加性注意力的性能优于点积注意力[(引用)](https://arxiv.org/abs/1703.03906)。我们推测，
# 对于较大的$d_k$值，点积的幅度会变得很大，将softmax函数推入梯度极小的区域(为说明点积
# 为什么会变大，假设$q$和$k$的分量是均值为$0$、方差为$1$的独立随机变量。那么它们的点积
# $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$的均值为$0$，方差为$d_k$)。为了抵消这种效果，
# 我们用$\frac{1}{\sqrt{d_k}}$对点积进行缩放。
#
#

# %% [markdown] id="bS1FszhVTsqG"
# ![](images/ModalNet-20.png)


# %% [markdown] id="TNtVyZ-pTsqH"
#
# 多头注意力使模型能够在不同位置同时关注来自不同表示子空间的信息。使用单个注意力头会
# 因为平均化而抑制这种能力。
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{其中}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# 其中投影是参数矩阵$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、
# $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、
# $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$和
# $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。
#
# 在本工作中，我们使用了$h=8$个并行的注意力层，即头。对于每个头，我们使用
# $d_k=d_v=d_{\text{model}}/h=64$。由于每个头的维度减小，总的计算成本与使用
# 完整维度的单头注意力相似。

# %% id="D2LBMKCQTsqH"
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


# %% [markdown] id="EDRba3J3TsqH"
# ### 我们模型中注意力机制的应用

# Transformer以三种不同的方式使用多头注意力:
# 1) 在"编码器-解码器注意力"层中,查询来自前一个解码器层,而记忆键和值来自编码器的输出。
# 这使得解码器的每个位置都能关注输入序列的所有位置。这模仿了序列到序列模型中典型的编码器-解码器
# 注意力机制,如[(引用)](https://arxiv.org/abs/1609.08144)。

# 2) 编码器包含自注意力层。在自注意力层中,所有的键、值和查询都来自同一个地方,在这种情况下,
# 是编码器中前一层的输出。编码器中的每个位置都可以关注编码器前一层的所有位置。

# 3) 类似地,解码器中的自注意力层允许解码器的每个位置关注解码器中直到该位置(包括该位置)的所有位置。
# 我们需要防止解码器中的信息向左流动以保持自回归特性。我们通过在缩放点积注意力中将softmax输入中
# 对应于非法连接的所有值屏蔽(设置为$-\infty$)来实现这一点。

# %% [markdown] id="M-en97_GTsqH"
# ## 位置前馈网络

# 除了注意力子层外,我们的编码器和解码器中的每一层都包含一个全连接前馈网络,
# 该网络分别且相同地应用于每个位置。这包括两个线性变换,中间有一个ReLU激活函数。

# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$

# 虽然不同位置的线性变换是相同的,但它们在不同层之间使用不同的参数。
# 另一种描述方式是将其视为核大小为1的两个卷积。输入和输出的维度是
# $d_{\text{model}}=512$,内层的维度是$d_{ff}=2048$。

# %% id="6HHCemCxTsqH"
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# %% [markdown] id="dR1YM520TsqH"
# ## 嵌入和Softmax
#
# 与其他序列转导模型类似，我们使用学习得到的嵌入将输入标记和输出标记转换为维度为$d_{\text{model}}$的向量。
# 我们还使用常规的学习得到的线性变换和softmax函数将解码器输出转换为预测下一个标记的概率。
# 在我们的模型中，两个嵌入层和softmax前的线性变换共享相同的权重矩阵，这类似于
# [(引用)](https://arxiv.org/abs/1608.05859)。在嵌入层中，我们将这些权重乘以$\sqrt{d_{\text{model}}}$。

# %% id="pyrChq9qTsqH"
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# %% [markdown] id="vOkdui-cTsqH"
# ## 位置编码
#
# 由于我们的模型不包含循环和卷积结构，为了让模型能够利用序列的顺序信息，我们必须
# 在序列中注入一些关于标记相对或绝对位置的信息。为此，我们在编码器和解码器堆栈的底部
# 为输入嵌入添加"位置编码"。位置编码具有与嵌入相同的维度$d_{\text{model}}$，
# 因此两者可以相加。位置编码有多种选择，包括可学习的和固定的
# [(引用)](https://arxiv.org/pdf/1705.03122.pdf)。
#
# 在本工作中，我们使用不同频率的正弦和余弦函数：
#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
#
# 其中$pos$是位置，$i$是维度。也就是说，位置编码的每个维度对应一个正弦曲线。
# 波长构成了从$2\pi$到$10000 \cdot 2\pi$的几何级数。我们选择这个函数是因为
# 我们假设它能让模型轻松学习关注相对位置，因为对于任何固定偏移量$k$，$PE_{pos+k}$
# 都可以表示为$PE_{pos}$的线性函数。
#
# 此外，我们对编码器和解码器堆栈中的嵌入和位置编码的和应用dropout。
# 对于基础模型，我们使用$P_{drop}=0.1$的比率。
#
#

# %% id="zaHGD4yJTsqH"
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# %% [markdown] id="EfHacTJLTsqH"
#
# > 下面的位置编码将基于位置添加正弦波。波的频率和偏移在每个维度上都不相同。

# %% id="rnvHk_1QokC6" type="example"
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)


# %% [markdown] id="g8rZNCrzTsqI"
#
# 我们还尝试使用学习得到的位置嵌入[(引用)](https://arxiv.org/pdf/1705.03122.pdf)，
# 发现两种版本产生了几乎相同的结果。我们选择正弦版本是因为它可能允许模型推广到比训练
# 期间遇到的序列更长的序列长度。

# %% [markdown] id="iwNKCzlyTsqI"
# ## 完整模型
#
# > 这里我们定义一个从超参数到完整模型的函数。

# %% id="mPe1ES0UTsqI"
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# %% [markdown]
# ## 推理:
#
# > 这里我们进行一个前向步骤来生成模型的预测。我们尝试使用我们的transformer来记忆输入。
# 由于模型还未经过训练，你会看到输出是随机生成的。在下一个教程中，我们将构建训练函数，
# 并尝试训练我们的模型来记忆1到10的数字。

# %%
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)


# %% [markdown]
# # 第二部分：模型训练

# %% [markdown] id="05s6oT9fTsqI"
# # 训练
#
# 本节描述我们模型的训练方案。

# %% [markdown] id="fTxlofs4TsqI"
#
# > 我们暂停一下,介绍一些训练标准编码器-解码器模型所需的工具。
# > 首先我们定义一个batch对象,用于在训练时保存源序列和目标序列,
# > 同时构建相应的掩码。

# %% [markdown] id="G7SkCenXTsqI"
# ## 批处理和掩码

# %%
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


# %% [markdown] id="cKkw5GjLTsqI"
#
# > 接下来我们创建一个通用的训练和评分函数来跟踪损失。我们传入一个通用的损失计算函数,
# > 该函数同时也处理参数更新。

# %% [markdown] id="Q8zzeUc0TsqJ"
# ## 训练循环

# %%
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


# %% id="2HAZD3hiTsqJ"
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# %% [markdown] id="aB1IF0foTsqJ"
# ## 训练数据和批处理
#
# 我们在标准的WMT 2014英德数据集上进行训练,该数据集包含约450万个句子对。
# 使用字节对编码(byte-pair encoding)对句子进行编码,共享的源语言-目标语言词表大小约为37000个词元。
# 对于英法翻译,我们使用了明显更大的WMT 2014英法数据集,其包含3600万个句子,
# 并将词元分割成32000个词片(word-piece)词表。
#
# 根据序列的近似长度将句子对组合成批次。每个训练批次包含一组句子对,
# 大约包含25000个源语言词元和25000个目标语言词元。

# %% [markdown] id="F1mTQatiTsqJ" jp-MarkdownHeadingCollapsed=true tags=[]
# ## 硬件和训练计划
#
# 我们在一台配备8个NVIDIA P100 GPU的机器上训练模型。对于使用论文中描述的超参数的基础模型,
# 每个训练步骤大约需要0.4秒。我们训练基础模型共100,000步,总计12小时。对于我们的大型模型,
# 每步训练时间为1.0秒。大型模型训练了300,000步(3.5天)。

# %% [markdown] id="-utZeuGcTsqJ"
# ## 优化器
#
# 我们使用Adam优化器[(引用)](https://arxiv.org/abs/1412.6980),
# 参数设置为$\beta_1=0.9$、$\beta_2=0.98$和$\epsilon=10^{-9}$。
# 在训练过程中,我们根据以下公式调整学习率:
#
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
#
# 这相当于在前$warmup\_steps$个训练步骤中线性增加学习率,之后按步数的平方根的倒数
# 比例减小学习率。我们使用$warmup\_steps=4000$。

# %% [markdown] id="39FbYnt-TsqJ"
#
# > 注意:这部分非常重要。需要使用这种设置来训练模型。

# %% [markdown] id="hlbojFkjTsqJ"
#
# > 这个模型在不同模型大小和优化超参数下的学习曲线示例。

# %% id="zUz3PdAnVg4o"
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# %% id="l1bnrlnSV8J5" tags=[]
def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


example_learning_schedule()


# %% [markdown] id="7T1uD15VTsqK"
# ## 正则化
#
# ### 标签平滑
#
# 在训练过程中，我们使用了值为$\epsilon_{ls}=0.1$的标签平滑技术
# [(引用)](https://arxiv.org/abs/1512.00567)。
# 这会降低困惑度，因为模型学会了更加不确定，但是
# 提高了准确率和BLEU分数。

# %% [markdown] id="kNoAVD8bTsqK"
#
# > 我们使用KL散度损失来实现标签平滑。我们不使用
# > one-hot目标分布，而是创建一个分布，使其对正确词有
# > `confidence`的置信度，剩余的`smoothing`质量
# > 分布在整个词表中。

# %% id="shU2GyiETsqK"
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# %% [markdown] id="jCxUrlUyTsqK"
#
# > 这里我们可以看到基于置信度的词语概率质量分布的示例。

# %% id="EZtKaaQNTsqK"
# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


show_example(example_label_smoothing)


# %% [markdown] id="CGM8J1veTsqK"
#
# > 标签平滑实际上会在模型对某个选择变得过于自信时开始惩罚模型。

# %% id="78EHzLP7TsqK"


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


show_example(penalization_visualization)


# %% [markdown] id="67lUqeLXTsqK"
# # 第一个示例
#
# > 我们可以从尝试一个简单的复制任务开始。给定一个来自小型词汇表的随机输入符号集，
# > 目标是生成回相同的符号。

# %% [markdown] id="jJa-89_pTsqK"
# ## 合成数据

# %% id="g1aTxeqqTsqK"
def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


# %% [markdown] id="XTXwD9hUTsqK"
# ## 损失计算

# %% id="3J8EJm87TsqK"
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


# %% [markdown] id="eDAI7ELUTsqL"
# ## 贪婪解码

# %% [markdown] id="LFkWakplTsqL" tags=[]
# > 为了简单起见,这段代码使用贪婪解码来预测翻译结果。
# %% id="N2UOpnT3bIyU"
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# %% id="qgIZ2yEtdYwe" tags=[]
# Train the simple copy task.


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)


# %% [markdown] id="OpuQv2GsTsqL"
# # 第三部分：真实世界示例
#
# > 现在我们来看一个使用Multi30k德英翻译任务的真实世界示例。
# > 这个任务比论文中考虑的WMT任务小得多，但它展示了整个
# > 系统的工作原理。我们还将展示如何使用多GPU处理来
# > 显著提升运行速度。

# %% [markdown] id="8y9dpfolTsqL" tags=[]
# ## 数据加载
#
# > 我们将使用torchtext和spacy进行数据集加载和
# > 分词处理。

# %%
# 加载spacy分词器模型，如果尚未下载则进行下载


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


# %% id="t4BszXXJTsqL" tags=[]
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# %% id="jU3kVlV5okC-" tags=[]


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])


# %% [markdown] id="-l-TFwzfTsqL"
#
# > 批处理对速度有很大影响。我们希望批次划分得非常均匀，
# > 同时将填充降到最低。为此，我们需要对默认的torchtext批处理
# > 进行一些修改。这段代码对默认的批处理进行了修补，以确保我们
# > 能搜索足够多的句子来找到紧凑的批次。

# %% [markdown] id="kDEj-hCgokC-" tags=[] jp-MarkdownHeadingCollapsed=true
# ## 迭代器

# %% id="wGsIHFgOokC_" tags=[]
def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


# %% id="ka2Ce_WIokC_" tags=[]
def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# %% [markdown] id="90qM8RzCTsqM"
# ## Training the System

# %%
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


# %% tags=[]
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if is_interactive_notebook():
    model = load_trained_model()


# %% [markdown] id="RZK_VjDPTsqN"
#
# > 训练完成后,我们可以对模型进行解码以生成一组翻译结果。这里我们仅翻译验证集中的第一个句子。
# > 由于这个数据集比较小,使用贪婪搜索的翻译结果也相当准确。

# %% [markdown] id="L50i0iEXTsqN"
# # 附加组件：BPE、搜索、平均

# %% [markdown] id="NBx1C2_NTsqN"
#
# > 到目前为止,我们主要介绍了Transformer模型本身。还有四个方面我们没有明确涉及。
# > 这些额外的功能都已在[OpenNMT-py](https://github.com/opennmt/opennmt-py)中实现。
#
#

# %% [markdown] id="UpqV1mWnTsqN"
#
# > 1) BPE/词片(Word-piece)：我们可以使用库来首先将数据预处理成子词单元。
# > 参见Rico Sennrich的[subword-nmt](https://github.com/rsennrich/subword-nmt)
# > 实现。这些模型会将训练数据转换成如下形式：

# %% [markdown] id="hwJ_9J0BTsqN"
# ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP
# ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

# %% [markdown] id="9HwejYkpTsqN"
#
# > 2) 共享嵌入：当使用共享词表的BPE时,我们可以在源语言/目标语言/生成器之间共享相同的权重向量。
# > 详见[(引用)](https://arxiv.org/abs/1608.05859)。要将此功能添加到模型中,只需执行以下操作：

# %% id="tb3j3CYLTsqN" tags=[]
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight


# %% [markdown] id="xDKJsSwRTsqN"
#
# > 3) 集束搜索(Beam Search)：这里过于复杂,不做详细介绍。可以参考
# > [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/)
# > 中的PyTorch实现。
# >
#

# %% [markdown] id="wf3vVYGZTsqN"
#
# > 4) 模型平均：论文通过对最后k个检查点取平均来创建集成效果。如果我们
# > 有多个模型,可以在训练后执行这个操作：

# %% id="hAFEa78JokDB"
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


# %% [markdown] id="Kz5BYJ9sTsqO"
# # 结果
#
# 在WMT 2014英德翻译任务中,大型Transformer模型(表2中的Transformer (big))
# 的表现超过了此前报告的最佳模型(包括集成模型)2.0以上的BLEU分数,
# 创造了28.4的新的SOTA(state-of-the-art) BLEU分数。该模型的配置列在表3的
# 最后一行。训练在8个P100 GPU上花费了3.5天。即使是我们的基础模型也超过了
# 所有已发表的模型和集成模型,而且训练成本只是其他竞争模型的一小部分。
#
# 在WMT 2014英法翻译任务中,我们的大型模型达到了41.0的BLEU分数,超过了
# 所有此前发表的单模型,训练成本不到之前最先进模型的1/4。用于英法翻译的
# Transformer (big)模型使用了dropout率Pdrop = 0.1,而不是0.3。
#

# %% [markdown]
# ![](images/results.png)

# %% [markdown] id="cPcnsHvQTsqO"
#
#
# > 使用上一节中的额外扩展,OpenNMT-py的复现在EN-DE WMT上达到了26.9。
# > 这里我已经将这些参数加载到我们的重新实现中。

# %%
# Load data and model for output checks


# %%
def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# execute_example(run_model_example)


# %% [markdown] id="0ZkkNTKLTsqO"
# ## 注意力可视化
#
# > 即使使用贪婪解码器，翻译的效果看起来也相当不错。我们
# > 可以进一步将其可视化，以观察注意力机制在每一层中发生了什么

# %%
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


# %% tags=[]
def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


# %% [markdown]
# ## Encoder Self Attention

# %% tags=[]
def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


show_example(viz_encoder_self)


# %% [markdown]
# ## Decoder Self Attention

# %% tags=[]
def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_self)


# %% [markdown]
# ## Decoder Src Attention

# %% tags=[]
def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_src)

# %% [markdown] id="nSseuCcATsqO"
# # 结论
#
# 希望这份代码能对未来的研究工作有所帮助。如果您遇到任何问题，
# 请随时与我们联系。
#
#
# 此致，
# Sasha Rush、Austin Huang、Suraj Subramanian、Jonathan Sum、Khalid Almubarak、
# Stella Biderman
