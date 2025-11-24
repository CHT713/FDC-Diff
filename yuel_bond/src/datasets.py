import os
import numpy as np
import pandas as pd
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const
from Bio.PDB import PDBParser
from pdb import set_trace

def parse_residues(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            # atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                # pocket_types.append(atom_type)
                pocket_types.append(residue_name)

    return {
        'coord': pocket_coords,
        'types': pocket_types,
    }

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule

# one hot for atoms
def atom_one_hot(atom):
    # n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    one_hot = np.zeros(n2)
    if atom not in const.ATOM2IDX:
        atom = 'Cl'
    one_hot[const.ATOM2IDX[atom]] = 1
    return one_hot

# one hot for amino acids
def aa_one_hot(residue):
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    one_hot = np.zeros(n1 + n2)
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot

def molecule_feat_mask():
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    mask = np.zeros(n1 + n2)
    mask[n1:] = 1
    return mask

def bond_one_hot(bond):
    one_hot = [0 for i in range(const.N_RDBOND_TYPES)]
    
    # Set the appropriate index to 1
    bond_type = bond.GetBondType()
    if bond_type in const.RDBOND2IDX:
        one_hot[const.RDBOND2IDX[bond_type]] = 1
    else:
        raise Exception('Unknown bond type {}'.format(bond_type))
        
    return one_hot

def parse_molecule(mol):
    atom_one_hots = []
    for atom in mol.GetAtoms():
        atom_one_hots.append(atom_one_hot(atom.GetSymbol()))

    # if mol has no conformer, positions is 0
    if mol.GetNumConformers() == 0:
        positions = np.zeros((mol.GetNumAtoms(), 3))
    else:
        positions = mol.GetConformer().GetPositions()

    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        one_hot = bond_one_hot(bond)
        bonds.append([i, j] + one_hot)
        bonds.append([j, i] + one_hot)

    return positions, np.array(atom_one_hots), np.array(bonds)

def parse_pocket(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            # atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                # pocket_types.append(atom_type)
                pocket_types.append(residue_name)

    pocket_one_hot = []
    for _type in pocket_types:
        pocket_one_hot.append(aa_one_hot(_type))
    pocket_one_hot = np.array(pocket_one_hot)

    return pocket_coords, pocket_one_hot

def get_pocket(mol, pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for ir,residue in enumerate(struct.get_residues()):
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(ir)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(axis=1) <= 6)[0]])

    return parse_pocket([r for (ir, r) in enumerate(struct.get_residues()) if ir in contact_residues])

import numpy as np

def pad_and_concatenate(tensor1, tensor2):
    N, a = tensor1.shape
    M, b = tensor2.shape
    
    # Pad tensor1 with zeros for the b columns it's missing
    tensor1_padded = np.pad(tensor1, 
                           pad_width=((0, 0), (0, b)),  # Pad b zeros on the right
                           mode='constant',
                           constant_values=0)
    
    # Pad tensor2 with zeros for the a columns it's missing
    tensor2_padded = np.pad(tensor2,
                           pad_width=((0, 0), (a, 0)),  # Pad a zeros on the left
                           mode='constant',
                           constant_values=0)
    
    # Concatenate along the first axis (stack vertically)
    return np.concatenate([tensor1_padded, tensor2_padded], axis=0)


# Mode to either predict:
# 1. has_bonds = True: only bond types for known connectivity
# 2. has_bonds = False: both connectivity and bond types
class BondDataset(Dataset):
    DATA_LIST_ATTRS = {
    'uuid', 'name', 'fragments_smi', 'linker_smi'
    }

    DATA_ATTRS_TO_PAD = {
        'positions', 'one_hot', 'edge_index', 'edge_attr', 'bond_orders', 'node_mask', 'edge_mask'
    }

    DATA_ATTRS_TO_ADD_LAST_DIM = {
        'node_mask', 'edge_mask',
    }

    def __init__(self, data=None, raw_data=None, data_path=None, prefix=None, save_path=None, device=None, has_bonds=False, progress_bar=True, noise=0):
        assert (data is not None) or (raw_data is not None) or all(x is not None for x in (data_path, prefix, device))
        
        self.distance_cutoff = 3
        self.has_bonds = has_bonds
        self.progress_bar = progress_bar
        self.noise = noise
        self.device = device
        self.save_path = save_path

        def set_data(data, save_path=None):
            self.data = data
            if save_path is not None:
                print(f'Saving dataset as {save_path}')
                torch.save(data, save_path)   
            return

        if data is not None:
            set_data(data, self.save_path)
            return

        if raw_data is not None:
            set_data(self.preprocess(raw_data, device), self.save_path)
            return

        suffix = '_bonds' if self.has_bonds else ''
        dataset_path = os.path.join(data_path, f'{prefix}{suffix}.pt')
            
        if os.path.exists(dataset_path) and self.noise == 0:
            print(f'Found dataset: {dataset_path}')
            self.data = torch.load(dataset_path, map_location=device)
        else:
            raw_data_path = os.path.join(data_path, f'{prefix}.pkl')
            print(f'Preprocessing dataset {raw_data_path}')
            with open(raw_data_path, 'rb') as f:
                raw_data = pickle.load(f)
            set_data(self.preprocess(raw_data, device), dataset_path if self.save_path is None else self.save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
        # if self.noise != 0:
        #     positions = d['positions']
        #     device = positions.device
        #     positions = positions.cpu().numpy()
        #     positions = positions + np.random.normal(0, self.noise, positions.shape)
        #     edge_index,edge_attr, bond_orders, edge_mask = self.get_edges(positions, d['bonds'].cpu().numpy())
        #     d['positions'] = torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device)
        #     d['edge_index'] = torch.tensor(np.array(edge_index), dtype=const.TORCH_INT, device=device)
        #     d['edge_attr'] = torch.tensor(np.array(edge_attr), dtype=const.TORCH_FLOAT, device=device)
        #     d['bond_orders'] = torch.tensor(np.array(bond_orders), dtype=const.TORCH_FLOAT, device=device)
        #     d['edge_mask'] = torch.tensor(np.array(edge_mask), dtype=const.TORCH_INT, device=device)
        # return d
    
    def get_edges(self, positions, bonds):
        bond_map = {(r[0],r[1]):r[2:] for r in bonds}

        n_bond_feats = len(const.RDKIT_BOND_TYPES)
        edge_index = []
        edge_attr = []
        bond_orders = []
        if self.has_bonds:
            for (i,j), bond_one_hot in bond_map.items():
                if not any(i==a and j==b for a,b in edge_index):
                    edge_index.append([i,j])
                    edge_attr.append([1])
                    bond_orders.append(bond_one_hot)
                if not any(j==a and i==b for a,b in edge_index):
                    edge_index.append([j,i])
                    edge_attr.append([1])
                    bond_orders.append(bond_one_hot)
        else:
            # only add bonds with distance <= distance_cutoff
            for i in range(len(positions)):
                for j in range(len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= self.distance_cutoff and i != j:
                        edge_index.append([i,j])
                        edge_attr.append([dist])

                        a = (i,j)
                        b = (j,i)
                        if a in bond_map:
                            bond_orders.append(bond_map[a])
                        elif b in bond_map:
                            bond_orders.append(bond_map[b])
                        else:
                            # the last bond_feat as 1 means no bond
                            bond_orders.append([0]*(n_bond_feats-1)+[1])

        edge_mask = np.ones(len(edge_index))
        return edge_index, edge_attr, bond_orders, edge_mask

    # @staticmethod
    def preprocess(self, raw_data, device):
        generator = tqdm(
            raw_data,
            total=len(raw_data)
        ) if self.progress_bar else raw_data
        data = []
        for row in generator:
            molecule_name = row['name']
            positions = row['positions'] # n_atoms 3
            one_hot = row['atoms'] # n_atoms node_features
            bonds = row['bonds'] # n_bonds 2+bond_features

            if self.noise != 0:
                positions = positions + np.random.normal(0, self.noise, positions.shape)

            edge_index, edge_attr, bond_orders, edge_mask = self.get_edges(positions, bonds)
            
            if len(positions) > 150 or len(positions) < 2 or len(bond_orders) == 0:
                tqdm.write(f'Skipping molecule {molecule_name} with {len(positions)} atoms')
                continue

            data.append({
                'name': molecule_name,
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'edge_index': torch.tensor(np.array(edge_index), dtype=const.TORCH_INT, device=device),
                'edge_attr': torch.tensor(np.array(edge_attr), dtype=const.TORCH_FLOAT, device=device),
                'bond_orders': torch.tensor(np.array(bond_orders), dtype=const.TORCH_FLOAT, device=device),
                'node_mask': torch.ones(len(positions), dtype=const.TORCH_INT, device=device),
                'edge_mask': torch.tensor(np.array(edge_mask), dtype=const.TORCH_INT, device=device),
                # 'bonds': torch.tensor(np.array(bonds), dtype=const.TORCH_FLOAT, device=device),
            })

        return data

def collate(batch):
    out = {}

    # collect the list attributes
    for data in batch:
        for key, value in data.items():
            if key in BondDataset.DATA_LIST_ATTRS or key in BondDataset.DATA_ATTRS_TO_PAD:
                out.setdefault(key, []).append(value)

    # pad the tensors
    for key, value in out.items():
        if key in BondDataset.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue

    # add last dimension to the tensor
    for key in BondDataset.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

