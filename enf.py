import numpy as np
import torch
from models.flow import Flow
atom_decoder = ['N/A', 'H', 'C', 'N', 'O', 'F']
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154 , 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}

bonds2 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 134, 'N': 129, 'O': 120, 'F': -1000},
          'N': {'H': -1000, 'C': 129, 'N': 125, 'O': 121, 'F': -1000},
          'O': {'H': -1000, 'C': 120, 'N': 121, 'O': 121, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}

bonds3 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 120, 'N': 116, 'O': 113, 'F': -1000},
          'N': {'H': -1000, 'C': 116, 'N': 110, 'O': -1000, 'F': -1000},
          'O': {'H': -1000, 'C': 113, 'N': -1000, 'O': -1000, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0

def check_stability(positions, atom_type, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[
                atom_type[j]]
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i
        if is_stable == False and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

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
        block_size=2,
        encoder_size=1
    )

    net.load_state_dict(
        torch.load("outputs/model_checkpoint_zxwjyjn3_10.pt", map_location="cpu")["model_state_dict"]
    )

    batch_size = 400
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

    # print(atoms_types.shape, atoms_types)
    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        
        atom_ty =atoms_types[idx, :size]
        pos_t = pos[idx, :size]

        if 0 in atom_ty or len(atom_ty) == 0:
            continue
        
        validity, _, _ = check_stability(pos_t, atom_ty, debug=True)
        if validity:
            valid += 1

    print(valid * 1.0 / batch_size)