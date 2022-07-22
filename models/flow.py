from torch import nn
import torch
from survae.transforms.bijections import ConditionalBijection, Bijection

from .block import ar_net_init, CouplingBlockFlow
from .fuse import FuseFlow

class Flow(nn.Module):
    def __init__(self, 
        hidden_dim=64,
        gnn_size=1,
        block_size=6,
        max_nodes=29,
        num_classes=6,
        encoder_size=4,
        no_constraint=False,
    ) -> None:

        super().__init__()

        self.transforms = nn.ModuleList([
            FuseFlow(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                gnn_size=gnn_size,
                encoder_size=encoder_size,
                context_dim=16,
                max_nodes=max_nodes,
                no_constraint=no_constraint
            )
        ])

        self.transforms += [
            CouplingBlockFlow(
                num_classes=num_classes,
                euclidean_dim=3,
                partition_size=3,
                max_nodes=max_nodes,
                no_constraint=no_constraint,
                ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size, num_classes=num_classes, euclidean_dim=3)
            ) for _ in range(block_size)
        ]

        self.transforms += [
            CouplingBlockFlow(
                num_classes=num_classes,
                euclidean_dim=3,
                partition_size=2,
                max_nodes=max_nodes, 
                no_constraint=no_constraint,
                ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size, num_classes=num_classes, euclidean_dim=3)
            ) for _ in range(2)
        ]

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