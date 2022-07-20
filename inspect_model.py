from models.flow import Flow
import torch
from torch.cuda.amp import GradScaler, autocast

net = Flow(
        hidden_dim=32,
        gnn_size=1,
        block_size=2,
        encoder_size=1
    )


states = torch.load("outputs/model_irregularity_zxwjyjn3_7_6062.pt", map_location="cpu")


net.load_state_dict(
    states["model_state_dict"]
)

input = states['input']
mask = states['mask']



xh, _ = net(input, mask=mask)


print(xh)