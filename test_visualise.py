from turtle import pos
from utils.visualise import plot_data3d
import matplotlib
import torch

if __name__ == "__main__":
    matplotlib.use('macosx')

    pos = torch.randn(1, 29, 3) * 2
    pos = pos.view(-1, 3)

    atom_types = torch.randint(0, 5, (1, 29)).squeeze().numpy()
    

    plot_data3d(
        positions=pos,
        atom_type=atom_types,
        spheres_3d=False
    )