import pytest
from rdkit import Chem
import sys
from pathlib import Path
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
import os
from rdkit.Chem import rdDetermineBonds
import uuid
# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent  # Goes up two levels from test.py
print(project_root)
sys.path.append(str(project_root))
# sys.path.append('../../')

import src
from src.molecule_builder import build_molecule, build_molecules
from src.datasets import parse_molecule, get_pocket, BondDataset, get_dataloader, collate
from src.metrics import is_valid, qed, sas
from src import const
from src.lightning import YuelBond
from src.visualizer import load_molecule_xyz
# test BondDataset with pytest
@pytest.fixture
def dataset():
    # use get_dataloader to get a dataloader; use collate to collate the data
    dataloader = get_dataloader(BondDataset(data_path='datasets', prefix='geom_val', device='cpu'), batch_size=2, shuffle=False, collate_fn=collate)
    return dataloader

def test_dataloader(dataset):
    # assert the keys
    for data in dataset:
        assert data['edge_mask'].shape[1] == data['edge_index'].shape[1]
        assert data['node_mask'].shape[1] == data['positions'].shape[1]
        assert 'edge_mask' in data
        assert 'edge_index' in data
        assert 'edge_attr' in data
        assert 'bond_orders' in data
        assert 'one_hot' in data
        assert 'positions' in data
        assert 'name' in data
        break


# test YuelBond with pytest
@pytest.fixture
def model():
    return YuelBond(data_path='datasets', train_data_prefix='geom_train', val_data_prefix='geom_val', torch_device='cpu', hidden_nf=128, n_layers=3)

def test_model(model):
    pass

# test forward of YuelBond
def test_forward(model, dataset):
    # run forward of model
    # dataset does not support indexing like dataset[0]
    # so we need to get the first batch
    data = next(iter(dataset))
    edge_true = data['bond_orders']
    edge_pred = model.forward(data)
    edge_mask = data['edge_mask']
    # assert the shape of edge_pred (batch size and last dimension)
    assert edge_pred.shape[0] == data['bond_orders'].shape[0]
    assert edge_pred.shape[-1] == len(const.RDKIT_BOND_TYPES)

    # assert the loss
    loss = model.loss_fn(edge_true, edge_pred, edge_mask)
    assert loss.shape == ()

# convert the smiles "O=C(NCCCc1ccccc1)NCCc1cccs1" to a molecule
# generate the 3D coordinates
# save the molecule to a xyz file
# test the function in "yuel_bond.py" with the xyz file
def test_yuel_bond():
    # convert the smiles to a molecule
    mol = Chem.MolFromSmiles("O=C(NCCCc1ccccc1)NCCc1cccs1")
    # save the molecule to a pdb file with a temporary random name
    # temp_pdb_path = f"{uuid.uuid4()}.pdb"
    # temp_sdf_path = f"{uuid.uuid4()}.sdf"
    temp_ipath = f"test.xyz"
    temp_opath = f"test.sdf"
    # use rdkit to generate the 3D coordinates and then write to the pdb file
    mol = Chem.AddHs(mol)
    # not 2d coordinates, rather 3d coordinates
    AllChem.EmbedMolecule(mol)  # Basic 3D embedding
    AllChem.MMFFOptimizeMolecule(mol)
    # write the molecule to a xyz file
    mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    Chem.MolToXYZFile(mol, temp_ipath)
    # assert load_molecule_xyz
    mol = load_molecule_xyz(temp_ipath)
    # delete hydrogens
    assert mol is not None
    # assert the number of atoms is correct
    assert mol.GetNumAtoms() == n_atoms

    # test the function in "yuel_bond.py" with the xyz file
    print("Running yuel_bond.py ...")
    os.system(f"python yuel_bond.py --input {temp_ipath} --output {temp_opath} --model models/geom_bs16_date18-04_time16-47-04.989651/last.ckpt")
    print("yuel_bond.py finished.")
    # assert the output file exists
    assert os.path.exists(temp_opath)
    # assert the output file has the correct bond orders
    mol_pred = Chem.MolFromMolFile(temp_opath, sanitize=False, removeHs=True)
    assert mol_pred is not None
    # assert the bond orders are correct
    for i in range(mol_pred.GetNumAtoms()):
        for j in range(i+1, mol_pred.GetNumAtoms()):
            assert mol_pred.GetBondBetweenAtoms(i, j).GetBondType() == mol.GetBondBetweenAtoms(i, j).GetBondType()
    # delete the temporary files
    # os.remove(temp_ipath)
    # os.remove(temp_opath)
