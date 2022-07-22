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

atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_decoder_int = [1, 6, 7, 8, 9]

def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

# 0.9419752371018524
if __name__ == "__main__":

    train_loader, test_loader = get_datasets(
        type="mqm9",
        batch_size=1000,
    )

    valid = 0
    total_num = 0
    valid_smiles = []
    for batch_data in train_loader:
        atoms_types = batch_data.x[:, :, 0:1].long().squeeze(2).numpy()
        pos = batch_data.pos
        mask = batch_data.mask

        
        for idx in range(atoms_types.shape[0]):
            total_num += 1
            size = mask[idx].to(torch.long).sum()
            atom_decoder_int = [1, 6, 7, 8, 9]
            atom_ty =[atom_decoder_int[i] for i in atoms_types[idx, :size]]

            pos_t = pos[idx, :size].tolist()

            try:
                mols = xyz2mol(
                    atom_ty,
                    pos_t,
                    use_huckel=True,
                    allow_charged_fragments=False,
                )


                for mol in mols:
                    smiles = Chem.MolToSmiles(mol)

                    if "." in smiles:
                        continue
                    else:
                        valid += 1
                        valid_smiles.append(smiles)
                        break
            except:
                pass

    pprint(valid_smiles)
    print(valid * 1.0 / total_num)