import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
    return x

class LipSwish_(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.swish = nn.SiLU(True)

    def forward(self, x):
        return self.swish(x).div_(1.1)

class FullyConnectedGNN(nn.Module):
    def __init__(
        self,
        in_dim = 3,
        out_dim = 6,
        m_dim = 16,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = True,
        m_pool_method = 'sum',
        soft_edges = False,
        activation = "SiLU"
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        # assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        edge_input_dim = in_dim * 2

        if activation == "LipSwish":
            self.activation = LipSwish_
        else:
            self.activation = getattr(nn, activation)
        
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            self.activation(),
            nn.Linear(edge_input_dim * 2, m_dim),
            self.activation()
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        ) if soft_edges else None

        # self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()
        self.node_norm = nn.LayerNorm(in_dim) if norm_feats else nn.Identity()
        
        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + m_dim, m_dim * 2),
            dropout,
            self.activation(),
            nn.Linear(m_dim * 2, out_dim),
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            # nn.init.normal_(module.weight, std = self.init_eps)
            nn.init.uniform_(module.weight, a = 0, b = 0.001)

    def forward(self, feats, mask = None):

        feats_j = rearrange(feats, 'b j d -> b () j d')
        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)
        
        edge_input = torch.cat((feats_i, feats_j), dim=-1)

        m_ij = self.edge_mlp(edge_input) # B X I X J X m_dim

        if exists(self.edge_gate):
            m_ij = m_ij * self.edge_gate(m_ij)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')
            mask_j = rearrange(mask, 'b j -> b () j')
            mask = mask_i * mask_j

        
        if exists(mask):
            m_ij_mask = rearrange(mask, '... -> ... ()')
            m_ij = m_ij.masked_fill(~m_ij_mask, 0.)
        
        if self.m_pool_method == 'mean':
            if exists(mask):
                # masked mean
                mask_sum = m_ij_mask.sum(dim = -2)
                m_i = safe_div(m_ij.sum(dim = -2), mask_sum)
            else:
                m_i = m_ij.mean(dim = -2)

        elif self.m_pool_method == 'sum':
            m_i = m_ij.sum(dim = -2)
        
        normed_feats = self.node_norm(feats)
        node_mlp_input = torch.cat((normed_feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input)
        
        return node_out
