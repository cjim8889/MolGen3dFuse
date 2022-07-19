from survae.transforms.bijections import Bijection
from .utils import create_mask_equivariant
from .coupling import MaskedCouplingFlow
from survae.transforms.bijections import Bijection

import torch
from torch import nn
from egnn_pytorch import EGNN
from .argmax.c_gnn import ModifiedEGNN

class ARNet(nn.Module):
    def __init__(self, hidden_dim=32, gnn_size=1, num_classes=6, euclidean_dim=3, idx=(0, 2)):
        super().__init__()

        self.num_classes = num_classes
        self.euclidean_dim = euclidean_dim
        self.idx = idx

        self.net = nn.ModuleList(
            [
                ModifiedEGNN(
                    dim=num_classes,
                    m_dim=hidden_dim, 
                    soft_edges=True,
                    coor_weights_clamp_value=1., 
                    num_nearest_neighbors=6, 
                    update_coors=True,
                    update_feats=True,
                    norm_feats=True,
                    norm_coors_constant=1.0,
                    norm_coors=True,
                    dropout=0.15,
                    m_pool_method='sum'
                ) for _ in range(gnn_size)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_classes + euclidean_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (self.idx[1] - self.idx[0]) * (num_classes + euclidean_dim) * 2),
        )

    def forward(self, x, mask=None):
        feats = x[:, :, :self.num_classes]
        coors = x[:, :, self.num_classes:]

        for net in self.net:
            feats, coors = net(feats, coors, mask=mask)
        
        z = torch.cat((feats, coors), dim=-1)

        if mask is None:
            z = torch.mean(z, dim=1) 
        else:
            z = torch.sum(z * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        
        z = self.mlp(z).view(x.shape[0], self.idx[1] - self.idx[0], (self.num_classes + self.euclidean_dim) * 2)
        z = nn.functional.pad(z, (0, 0, self.idx[0], 29 - self.idx[1], 0, 0), 'constant', 0)
        return z

def ar_net_init(hidden_dim=128, gnn_size=2, num_classes=6, euclidean_dim=3):
    def _init(idx):
        return ARNet(hidden_dim=hidden_dim, gnn_size=gnn_size, euclidean_dim=euclidean_dim, num_classes=num_classes, idx=idx)

    return _init

class CouplingBlockFlow(Bijection):
    def __init__(self,
        num_classes=6,
        euclidean_dim=3,
        ar_net_init=ar_net_init(hidden_dim=64, num_classes=6, gnn_size=1),
        mask_init=create_mask_equivariant,
        max_nodes=29,
        partition_size=2
    ):
        
        super(CouplingBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, partition_size):
            ar_net = ar_net_init((idx, min(idx + partition_size, max_nodes)))
            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)
            
            tr = MaskedCouplingFlow(ar_net, mask=mask, last_dimension=num_classes + euclidean_dim, split_dim=-1)
            
            self.transforms.append(tr)

    def forward(self, x, mask=None, logs=None):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            if isinstance(transform, Bijection):
                x, ldj = transform(x, mask=mask, logs=logs)

            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, mask=None):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], Bijection):
                z, ldj = self.transforms[idx].inverse(z, mask=mask)

            log_prob += ldj
        
        return z, log_prob