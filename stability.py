from json import encoder
from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

from models.flow import Flow

import torch
import numpy as np
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
atom_decoder_int = [0, 1, 6, 7, 8, 9]

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


if __name__ == "__main__":
    net = Flow(
        hidden_dim=32,
        gnn_size=1,
        block_size=4,
        encoder_size=2
    )

    net.load_state_dict(
        torch.load("outputs/model_irregularity_31gk3i0z_4_3747.pt", map_location="cpu")["model_state_dict"]
    )

    batch_size = 1000
    mask = torch.ones(batch_size, 29).to(torch.bool)
    mask_size = torch.randint(3, 29, (batch_size,))
    
    for idx in range(batch_size):
        mask[idx, mask_size[idx]:] = False

    categorical = torch.randn(batch_size, 29, 6)
    continuous = remove_mean_with_mask(torch.randn(batch_size, 29, 3), mask)

    z = torch.cat((categorical * mask.unsqueeze(2), continuous), dim=-1)

    with torch.no_grad():
        xh, _ = net.inverse(z, mask=mask)

    atoms_types, pos = torch.split(xh, [1, 3], dim=-1)
    
    valid = 0
    atoms_types = atoms_types.long().squeeze(2).numpy()
    pos = pos.numpy()

    valid_smiles =[]
    valid_mols = []

    # print(atoms_types.shape, atoms_types)
    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        atom_decoder_int = [0, 1, 6, 7, 8, 9]
        atom_ty =[atom_decoder_int[i] for i in atoms_types[idx, :size]]

        pos_t = pos[idx, :size].tolist()

        if 0 in atom_ty or len(atom_ty) == 0:
            continue

        try:
            mols = xyz2mol(
                atom_ty,
                pos_t,
                use_huckel=True,
            )


            for mol in mols:
                smiles = Chem.MolToSmiles(mol)

                # if "." in smiles:
                    # continue
                # else:
                valid += 1
                valid_smiles.append(smiles)
                valid_mols.append(mol)
                break
        except:
            pass

    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)