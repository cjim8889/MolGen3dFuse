import torch

from models.argmax.c_gnn import FullyConnectedGNN



net = FullyConnectedGNN(
    in_dim=6, 
    out_dim=6,
    m_dim=32,
    soft_edges=True
)


feats = torch.randn(1, 29, 6)

out = net(feats)

print(out)