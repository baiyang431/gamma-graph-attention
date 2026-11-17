"""手写多头图注意力；仅依赖 PyTorch。"""
import torch
from torch import nn
from torch.nn import functional as F


def add_self_loops(edges, num_nodes):
    edges = edges[:, edges[0] != edges[1]]
    nodes = torch.arange(num_nodes, device=edges.device)
    return torch.cat((edges, torch.stack((nodes, nodes))), dim=1)


def neighbor_softmax(scores, dst, num_nodes):
    index = dst[:, None].expand_as(scores)
    maxima = scores.new_full((num_nodes, scores.shape[1]), -float("inf"))
    # 平移量不参与求导；softmax 对每个邻域的共同平移不变。
    maxima.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
    values = (scores - maxima[dst]).exp()
    totals = scores.new_zeros((num_nodes, scores.shape[1]))
    totals.index_add_(0, dst, values)
    return values / totals[dst]


class GAT(nn.Module):
    """同构有向多重图注意力；edges 为 [source, destination]，自环由调用方处理。"""

    def __init__(self, in_feats, out_feats, heads=1, concat=True, attention=True):
        super().__init__()
        if min(in_feats, out_feats, heads) < 1:
            raise ValueError("特征维度与头数必须为正数")
        self.heads = heads
        self.out_feats = out_feats
        self.concat = concat
        self.attention = attention
        self.fc = nn.Linear(in_feats, heads * out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads * out_feats))
        if attention:
            self.attn_l = nn.Parameter(torch.empty(1, heads, out_feats))
            self.attn_r = nn.Parameter(torch.empty(1, heads, out_feats))
            nn.init.xavier_normal_(self.attn_l)
            nn.init.xavier_normal_(self.attn_r)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x, edges, return_attention=False):
        if x.ndim != 2 or x.shape[1] != self.fc.in_features:
            raise ValueError("x 必须为 [N, in_feats]")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError("edges 必须为 [2,E] int64")
        if edges.device != x.device:
            raise ValueError("edges 和 x 必须位于同一设备")
        if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError("边索引越界")
        if not torch.isfinite(x).all():
            raise ValueError("节点特征包含 NaN 或无穷值")
        src, dst = edges
        z = self.fc(x).reshape(len(x), self.heads, self.out_feats)
        if self.attention:
            left = (z * self.attn_l).sum(-1)
            right = (z * self.attn_r).sum(-1)
            scores = F.leaky_relu(left[src] + right[dst], negative_slope=0.2)
        else:
            scores = z.new_zeros((len(src), self.heads))
        alpha = neighbor_softmax(scores, dst, len(x))
        messages = z[src] * alpha.unsqueeze(-1)
        out = z.new_zeros((len(x), self.heads, self.out_feats))
        out.index_add_(0, dst, messages)
        out = out + self.bias.reshape(1, self.heads, self.out_feats)
        out = out.flatten(1) if self.concat else out.mean(1)
        return (out, alpha) if return_attention else out
