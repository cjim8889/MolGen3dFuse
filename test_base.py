import torch
from torch import nn
from models.argmax.surjectives import ArgmaxSurjection, ContextNet
from models.argmax.conditionals import MaskedConditionalNormal


hidden_dim = 16
context_dim = 16
num_classes = 6

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


mask = torch.ones(1, 29, dtype=torch.bool)
mask[:, -2:] = False


context = torch.randint(0, 6, (1, 29))
context = context_net(context)


z, log_p = encoder_base.sample_with_log_prob(context=context, mask=mask)
print(z, log_p)