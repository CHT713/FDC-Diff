import torch
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import random

from sklearn.decomposition import PCA
from src import const
from rdkit import Chem
from rdkit.Chem import AllChem


def save_xyz_file(path, one_hot, positions, node_mask, names, suffix=''):
    idx2atom = const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        f = open(os.path.join(path, f'{names[batch_i]}_{suffix}.xyz'), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()

def load_xyz_files(path, suffix=''):
    files = []
    for fname in os.listdir(path):
        if fname.endswith(f'_{suffix}.xyz'):
            files.append(fname)
    files = sorted(files, key=lambda f: -int(f.replace(f'_{suffix}.xyz', '').split('_')[-1]))
    return [os.path.join(path, fname) for fname in files]


def load_molecules_xyz(file):
    """
    Read an XYZ file and convert it to an RDKit molecule without adding bonds.
    Returns only the RDKit molecule.
    """
    mols = []
    # Read the XYZ file
    with open(file, encoding='utf8') as f:
        # while the end of the file is not reached
        while True:
            line = f.readline()
            if line == '':
                break
            elif line.strip() == '':
                continue
            
            n_atoms = int(line)
            f.readline()  # Skip comment line
            # read until the whole line is a number
            atoms = [f.readline() for _ in range(n_atoms)]
    
            # Create a new RDKit molecule
            mol = Chem.RWMol()
            
            # Add atoms to the molecule
            for i, atom in enumerate(atoms):
                atom_data = atom.split()
                atom_type = atom_data[0]
                
                # Add atom to RDKit molecule
                rdkit_atom = Chem.Atom(atom_type)
                mol.AddAtom(rdkit_atom)
            
            # Convert to regular molecule (not editable)
            mol = mol.GetMol()
            
            # Add 3D coordinates to the molecule
            conf = Chem.Conformer(n_atoms)
            for i, atom in enumerate(atoms):
                atom_data = atom.split()
                position = [float(e) for e in atom_data[1:]]
                conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                    position[0],
                    position[1],
                    position[2]
                ))
            mol.AddConformer(conf)

            mols.append(mol)
    
    return mols




