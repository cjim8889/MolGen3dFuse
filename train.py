import argparse
from utils import FlowExp
import torch
from torch import nn

parser = argparse.ArgumentParser(description="Molecular Generation MSc Project: 3D")
parser.add_argument("--type", help="Type of experiments e.g. argmaxadj")
parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)

parser.add_argument("--hidden_dim", help="Hidden dimension", type=int, default=64)
parser.add_argument("--block_size", help="Block length t parameter", type=int, default=12)
parser.add_argument("--gnn_size", help="Gnn size", type=int, default=2)
parser.add_argument("--base", help="Base distribution", type=str, default="standard")
parser.add_argument("--encoder_size", help="Encoder size", type=int, default=2)

parser.add_argument("--optimiser", help="Optimiser", type=str, default="Adam")
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-03)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-06)

parser.add_argument("--scheduler", help="Scheduler", type=str, default="StepLR")
parser.add_argument("--scheduler_step", help="Scheduler step", type=int, default=3)
parser.add_argument("--scheduler_gamma", help="Scheduler gamma", type=float, default=0.96)

parser.add_argument("--upload", help="Upload to wandb", type=bool, default=False)
parser.add_argument("--upload_interval", help="Upload to wandb every n epochs", type=int, default=10)

parser.add_argument("--autocast", help="Autocast", type=int, default=0)
parser.add_argument("--loadfrom", help="Load from checkpoint", type=str, default=None)
parser.add_argument("--no_opt", help="No optimiser", type=int, default=0)
parser.add_argument("--no_constraint", help="No constraint", type=int, default=0)

# parser.add_argument("--encoder_size", help="Encoder Size for Vert Net", type=int, default=2)
parser.add_argument("--classifier", help="Classifier", type=str, default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, nn.LazyLinear):
        return
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0, 0.001)

if __name__ == "__main__":
    args = parser.parse_args()

    exp = None

    if args.type  == "flow":
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
            hidden_dim=args.hidden_dim,
            block_size=args.block_size,
            gnn_size=args.gnn_size,
            base=args.base,
            loadfrom=args.loadfrom,
            autocast=args.autocast != 0,
            no_opt=args.no_opt == 0,
            no_constraint=args.no_constraint == 1,
            encoder_size=args.encoder_size,
        )

        exp = FlowExp(config=config)
    
    exp.train()