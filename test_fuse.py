import torch
from models.fuse import FuseFlow
from models.block import CouplingBlockFlow, ar_net_init

net = FuseFlow(
    num_classes=6,
    hidden_dim=64,
    gnn_size=2,
    encoder_size=2,
    context_dim=16,
    max_nodes=29
)

block = CouplingBlockFlow(
    num_classes=6,
    euclidean_dim=3,
    max_nodes=29,
    partition_size=2,
    ar_net_init=ar_net_init(hidden_dim=64, gnn_size=2)
)

catagorical = torch.randint(0, 6, (1, 29, 1), dtype=torch.float)
continuous = torch.randn(1, 29, 3)


x = torch.cat([catagorical, continuous], dim=-1)

z, _ = net(x)
q, _ = block(z)
# print(out)
print(q)
# x_re, _ = net.inverse(z)
# print(x_re)
