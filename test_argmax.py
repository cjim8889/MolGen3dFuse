from models.argmax.surjectives import ArgmaxSurjection, ContextNet
from models.argmax.conditionals import ConditionalBlockFlow, ar_net_init, MaskedConditionalInverseFlow, MaskedConditionalNormal
import torch
from torch import nn

num_classes = 5
hidden_dim = 64
context_dim = 16
encoder_size = 4

context_net = ContextNet(
    hidden_dim=hidden_dim,
    context_dim=context_dim,
    num_classes=num_classes
)

encoder_base = MaskedConditionalNormal(
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
        ar_net_init=ar_net_init(hidden_dim=hidden_dim, num_classes=num_classes, gnn_size=2, context_dim=context_dim),
    ) for _ in range(encoder_size)
]
inverse = MaskedConditionalInverseFlow(
    encoder_base, 
    transforms=transforms,
    context_init=context_net
)

surjection = ArgmaxSurjection(inverse, num_classes=num_classes)

feats = torch.randint(0, 5, (1, 29))
mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

z, log_p = surjection.forward(feats, mask=mask)

print(log_p)

# z_p, log_p_p = surjection.forward_new(feats)
# print(log_p_p)