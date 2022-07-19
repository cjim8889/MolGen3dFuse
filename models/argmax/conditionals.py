from survae.transforms.bijections import ConditionalBijection
from survae.utils import sum_except_batch
from ..utils import create_mask_equivariant
from .c_gnn import FullyConnectedGNN
import torch
from torch import nn

class MaskedConditionalCouplingFlow(ConditionalBijection):
    def __init__(self, ar_net, mask, split_dim=-1, last_dimension=6):
        super(MaskedConditionalCouplingFlow, self).__init__()
        
        self.ar_net = ar_net

        self.register_buffer("mask", mask.unsqueeze(2))
        self.scaling_factor = nn.Parameter(torch.zeros(last_dimension))
        self.split_dim = split_dim

    def forward(self, x, context):
        return self._transform(x, context, forward=True)

    def inverse(self, z, context):
        return self._transform(z, context, forward=False)

    def _transform(self, z, context, forward=True):
        z_masked = z * self.mask
        alpha, beta = self.ar_net(z_masked, context).chunk(2, dim=self.split_dim)

        # scaling factor idea inspired by UvA github to stabilise training 
        scaling_factor = self.scaling_factor.exp().view(1, 1, -1)

        alpha = torch.tanh(alpha / scaling_factor) * scaling_factor

        alpha = alpha * ~self.mask
        beta = beta * ~self.mask
        
        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            log_det = sum_except_batch(alpha)
        else:
            z = (z * torch.exp(-alpha)) - beta
            log_det = -sum_except_batch(alpha)
        
        return z, log_det

class ConditionalARNet(nn.Module):
    def __init__(self, num_classes=6, context_dim=16, hidden_dim=64, gnn_size=2, idx=(0, 2)):
        super().__init__()
        
        self.num_classes = num_classes
        self.idx = idx
        self.net = nn.ModuleList([
            FullyConnectedGNN(
                in_dim=num_classes + context_dim,
                out_dim=num_classes + context_dim,
                m_dim=hidden_dim,
                norm_feats=True,
                soft_edges=True
            ) for _ in range(gnn_size)
        ])

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear((self.idx[1] - self.idx[0]) * num_classes * 2),
        )
    # x: B x 9 x 6
    # context: B x 9 x 6
    def forward(self, x, context):
        z = torch.cat((x, context), dim=-1)
        
        for gnn in self.net:
            z = gnn(z)


        z = torch.mean(z, dim=1)
        z = self.mlp(z).view(x.shape[0], self.idx[1] - self.idx[0], self.num_classes * 2)
        z = nn.functional.pad(z, (0, 0, self.idx[0], 29 - self.idx[1], 0, 0), 'constant', 0)
        
        return z

def ar_net_init(num_classes=6, context_dim=16, hidden_dim=64, gnn_size=2):
    def create(idx):
        return ConditionalARNet(num_classes=num_classes, context_dim=context_dim, hidden_dim=hidden_dim, gnn_size=gnn_size, idx=idx)

    return create

class ConditionalBlockFlow(ConditionalBijection):
    def __init__(self, ar_net_init=ar_net_init(hidden_dim=64),
            max_nodes=9,
            num_classes=6,
            partition_size=1,
            mask_init=create_mask_equivariant,
            split_dim=-1):

        super(ConditionalBlockFlow, self).__init__()
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, partition_size):
            ar_net = ar_net_init((idx, min(idx + partition_size, max_nodes)))
            mask = mask_init([i for i in range(idx, min(idx + partition_size, max_nodes))], max_nodes)

            tr = MaskedConditionalCouplingFlow(ar_net, mask=mask, split_dim=split_dim, last_dimension=num_classes)
            self.transforms.append(tr)
        
       
    def forward(self, x, context):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x, context)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context):
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z, context)
            log_prob += ldj
        
        return z, log_prob