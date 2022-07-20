from .argmax import ArgmaxSurjection
from survae.transforms.bijections import Bijection
from .argmax import ConditionalBlockFlow, ar_net_init, ArgmaxSurjection, ContextNet
from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
import torch
from torch import nn

class FuseFlow(Bijection):
    def __init__(self, 
        num_classes=6, 
        hidden_dim=64, 
        gnn_size=2, 
        encoder_size=2,
        context_dim=16,
        euclidean_dim=3,
        max_nodes=29) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.euclidean_dim = euclidean_dim

        context_net = ContextNet(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_classes=num_classes
        )

        encoder_base = ConditionalNormal(
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes * 2),
            ),
            split_dim=-1
        )

        transforms = [
            ConditionalBlockFlow(
                max_nodes=max_nodes,
                num_classes=num_classes,
                partition_size=1,
                ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=gnn_size, context_dim=context_dim),
            ) for _ in range(encoder_size)
        ]


        inverse = ConditionalInverseFlow(
            encoder_base, 
            transforms=transforms,
            context_init=context_net
        )

        self.surjection = ArgmaxSurjection(inverse, num_classes=num_classes)


    def forward(self, x, mask=None, logs=None):
        categorical, continuous = torch.split(x, (1, self.euclidean_dim), dim=-1)

        z_cat, log_p_cat = self.surjection(categorical.long().squeeze(2))

        out = torch.cat([z_cat, continuous], dim=-1)

        return out, log_p_cat

    def inverse(self, z, mask=None, logs=None):
        categorical, continuous = torch.split(z, (self.num_classes, self.euclidean_dim), dim=-1)

        z_cat, log_p_cat = self.surjection.inverse(categorical)

        out = torch.cat([z_cat.unsqueeze(2), continuous], dim=-1)

        return out, log_p_cat