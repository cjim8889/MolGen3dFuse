import torch
from models.argmax.conditionals import ConditionalARNet, ConditionalBlockFlow, ar_net_init


# net = ConditionalARNet(
#     num_classes=6,
#     context_dim=16,
#     hidden_dim=64,
#     gnn_size=2,
#     idx=(0, 2)
# )

net = ConditionalBlockFlow(
    ar_net_init=ar_net_init(hidden_dim=64, context_dim=16, gnn_size=2),
    max_nodes=29,
    num_classes=6,
    partition_size=1,
)

feats = torch.randn(1, 29, 6)
context = torch.randn(1, 29, 16)


out = net(feats, context)
print(out)