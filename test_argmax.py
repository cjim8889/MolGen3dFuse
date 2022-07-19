from models.argmax.surjectives import ArgmaxSurjection, ContextNet
from models.argmax.conditionals import ConditionalBlockFlow, ar_net_init
from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
import torch
from torch import nn

num_classes = 6
hidden_dim = 64
context_dim = 16
encoder_size = 4

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
        max_nodes=29,
        num_classes=num_classes,
        partition_size=1,
        ar_net_init=ar_net_init(hidden_dim=hidden_dim, gnn_size=2, context_dim=context_dim),
    ) for _ in range(encoder_size)
]
inverse = ConditionalInverseFlow(
    encoder_base, 
    transforms=transforms,
    context_init=context_net
)

surjection = ArgmaxSurjection(inverse, num_classes=6)

feats = torch.randint(0, 6, (1, 29))

print(feats)
z, log_p = surjection(feats)

print(z)

feats_r, _ = surjection.inverse(z)
print(feats_r)