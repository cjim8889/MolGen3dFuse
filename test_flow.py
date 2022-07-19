from models.flow import Flow
import torch


catagorical = torch.randint(0, 6, (1, 29, 1), dtype=torch.float)
continuous = torch.randn(1, 29, 3)


x = torch.cat([catagorical, continuous], dim=-1)

mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False

net = Flow(
    hidden_dim=64,
    gnn_size=2,
    block_size=2,
    max_nodes=29,
    num_classes=6,
    encoder_size=2
)


out, _ = net(x, mask=mask)

x_re, _ = net.inverse(out, mask=mask)



