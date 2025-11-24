import argparse
import os
import numpy as np

import torch
import subprocess
from rdkit import Chem
from Bio.PDB import PDBParser

from src import const
from src.datasets import (
    collate, get_dataloader, BondDataset, parse_residues, parse_pocket, parse_molecule
)
from src.lightning import YuelBond
from src.visualizer import save_xyz_file, load_molecules_xyz
from src.utils import FoundNaNException, set_deterministic
from tqdm import tqdm
import yaml
from rdkit.Chem import Atom, BondType
from rdkit.Geometry import Point3D

parser = argparse.ArgumentParser()
parser.add_argument(
    'input', action='store', type=str,
    help='Path to the input molecule file (.xyz or .pdb)'
)
parser.add_argument(
    'output', action='store', nargs='?', type=str, default=None,
    help='Directory where output molecule will be saved'
)
parser.add_argument(
    '--dataset', action='store_true',
    help='Whether the input is a dataset'
)
parser.add_argument(
    '--model', action='store', type=str, required=False,default=None,
    help='Path to the YuelBond model'
)
parser.add_argument(
    '--save_filtered', action='store', type=str, required=False, default=None,
    help='Path to the filtered molecules'
)
parser.add_argument(
    '--random_seed', action='store', type=int, required=False, default=None,
    help='Random seed'
)
parser.add_argument(
    '--distance_cutoff', action='store', type=float, required=False, default=3.0,
    help='Distance cutoff for bond prediction (in Angstroms)'
)
parser.add_argument(
    '--batch_size', action='store', type=int, required=False, default=1,
    help='Batch size for processing'
)
# mode: 2d, 3d
parser.add_argument(
    '--mode', action='store', type=str, required=False, default='3d',
    help='Whether to include bonds in the input molecule'
)
parser.add_argument(
    '--kekulize', action='store_true',
    help='Whether to kekulize the input molecule'
)

def read_sdf(path, removeHs=True, generator=False):
    """
    Custom SDF reader that processes the file in chunks as specified:
    1. Reads 3 lines (header)
    2. Reads counts line (atoms and bonds)
    3. Reads atom block
    4. Reads bond block
    5. Reads until '$$$$' delimiter
    6. Yields RDKit molecule
    """
    with open(path, 'r') as f:
        bar = tqdm(desc="Reading SDF file")
        while True:
            # Read and skip header (3 lines)
            for _ in range(3):
                if not f.readline():  # EOF
                    return
            
            # Read counts line
            counts_line = f.readline()
            if not counts_line:  # EOF
                return
            
            try:
                num_atoms = int(counts_line[:3].strip())
                num_bonds = int(counts_line[3:6].strip())
            except ValueError:
                raise ValueError("Invalid counts line in SDF file")
            
            # Create new molecule and conformer
            mol = Chem.RWMol()
            atom_indices = []  # Track original atom indices for bond creation
            coords = []  # Store coordinates for remaining atoms
            
            # Read atom block
            for atom_idx in range(num_atoms):
                line = f.readline()
                if not line:  # EOF
                    raise ValueError("Unexpected EOF while reading atoms")
                
                # Parse atom information (positions fixed in SDF format)
                x = float(line[0:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
                symbol = line[31:34].strip().capitalize()
                
                if (removeHs and symbol == 'H') or symbol == 'D' or symbol == 'X':
                    atom_indices.append(None)  # Mark as deleted
                    continue
                
                try:
                    atom = Chem.Atom(symbol)
                except:
                    print(f"Error adding atom {symbol} to molecule")
                    continue
                
                new_idx = mol.AddAtom(atom)
                atom_indices.append(new_idx)
                coords.append((x, y, z))
            
            # Add conformer to molecule
            conf = Chem.Conformer(len(coords))
            for i, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(i, Point3D(x, y, z))
            mol.AddConformer(conf)
            
            # Read bond block - only if we have atoms left after potential H removal
            if mol.GetNumAtoms() > 0:
                for _ in range(num_bonds):
                    line = f.readline()
                    if not line:  # EOF
                        raise ValueError("Unexpected EOF while reading bonds")
                    
                    # Parse bond info (positions fixed in SDF format)
                    atom1_orig = int(line[0:3].strip()) - 1  # Convert to 0-based
                    atom2_orig = int(line[3:6].strip()) - 1
                    bond_type_code = int(line[6:9].strip())
                    
                    # Skip bonds involving removed atoms (especially Hs)
                    if (atom1_orig >= len(atom_indices) or 
                        atom2_orig >= len(atom_indices) or
                        atom_indices[atom1_orig] is None or 
                        atom_indices[atom2_orig] is None):
                        continue
                    
                    # Get new atom indices after potential H removal
                    atom1_idx = atom_indices[atom1_orig]
                    atom2_idx = atom_indices[atom2_orig]
                    
                    # Convert bond type code to RDKit bond type
                    if bond_type_code == 1:
                        bond_type = BondType.SINGLE
                    elif bond_type_code == 2:
                        bond_type = BondType.DOUBLE
                    elif bond_type_code == 3:
                        bond_type = BondType.TRIPLE
                    elif bond_type_code == 4:
                        bond_type = BondType.AROMATIC
                    else:
                        bond_type = BondType.SINGLE  # Default to single if unknown
                    
                    # Add bond
                    mol.AddBond(atom1_idx, atom2_idx, bond_type)
            
            # Skip properties until '$$$$'
            while True:
                line = f.readline()
                if not line:  # EOF
                    raise ValueError("Missing '$$$$' delimiter in SDF file")
                if line.strip() == '$$$$':
                    break
            
            mol = mol.GetMol()
            bar.update(1)
            yield mol

def read_molecules(path):
    if path.endswith('.pdb'):
        raise Exception('PDB files are not supported')
    elif path.endswith('.mol'):
        raise Exception('MOL files are not supported')
    elif path.endswith('.mol2'):
        raise Exception('MOL2 files are not supported')
    elif path.endswith('.sdf'):
        # return Chem.SDMolSupplier(path, sanitize=False, removeHs=True, strictParsing=False)
        return list(read_sdf(path))
    elif path.endswith('.xyz'):
        # For .xyz files, we'll use the load_molecule_xyz function from visualizer.py
        return load_molecules_xyz(path)
    raise Exception('Unknown file extension')

def create_molecule_from_predictions(positions, one_hot, edge_index, edge_pred, node_mask, edge_mask, name=None):
    """Create a molecule using the predicted bond types."""
    # positions:  (1, n_nodes, 3)
    # one_hot:    (1, n_nodes, n_feats)
    # edge_index: (1, n_edges, 2)
    # edge_pred:  (1, n_edges, n_bond_types)
    # node_mask:  (1, n_nodes, 1)
    # edge_mask:  (1, n_edges, 1)
    # name:       (1,)
    idx2atom = const.IDX2ATOM
    idx2bond = const.IDX2RDBOND

    node_mask = node_mask.squeeze(-1)
    edge_mask = edge_mask.squeeze(-1)
    
    # Create a new molecule
    mol = Chem.RWMol()
    
    # Add atoms
    atom_types = torch.argmax(one_hot.squeeze(0), dim=1)
    for i in range(positions.size(1)):
        if node_mask[0, i] == 1:  # Only add atoms that are not masked
            atom_type = idx2atom[atom_types[i].item()]
            atom = Chem.Atom(atom_type)
            mol.AddAtom(atom)
    
    # Add bonds based on predictions
    # Convert edge_pred to bond types
    bond_types = torch.argmax(edge_pred.squeeze(0), dim=1)
    bond_probs = torch.softmax(edge_pred.squeeze(0), dim=1)  # (n_edges, n_bond_types)
    
    # Add bonds
    for i in range(edge_index.size(1)):
        if edge_mask[0, i] == 0:
            continue

        bond_type_idx = bond_types[i].item()
        bond_type = idx2bond[bond_type_idx]
        if bond_type == Chem.rdchem.BondType.ZERO:
            continue

        # Get the atoms connected by this edge
        atom1_idx = edge_index[0, i, 0].item()
        atom2_idx = edge_index[0, i, 1].item()

        # print(edge_index[0,i])
        # print(atom1_idx, atom2_idx)

        # skip if the bond has already been added
        if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
            continue
        
        # Skip if either atom is masked
        if node_mask[0, atom1_idx] == 0 or node_mask[0, atom2_idx] == 0:
            continue
        
        # Add the bond
        mol.AddBond(atom1_idx, atom2_idx, bond_type)

        bond_obj = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond_obj is not None:
            # Format: "prob_single,prob_double,prob_triple,..."
            prob_str = ",".join([f"{const.BOND_TYPE_NAMES[ip]}:{p:.4f}" for ip,p in enumerate(bond_probs[i].tolist())])
            bond_obj.SetProp("probs", prob_str)
    
    # Add 3D coordinates
    conformer = Chem.Conformer()
    for i in range(positions.size(1)):
        if node_mask[0, i] == 1:  # Only add coordinates for atoms that are not masked
            x, y, z = positions[0, i].tolist()
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conformer)

    if name is not None:
        mol.SetProp('_Name', name)
    
    return mol

def process_folder(input_path, output_path, model=None, distance_cutoff=3.0, mode='3d', kekulize=False, device=None, save_filtered=None):
    if output_path is None:
        output_path = input_path
    
    # create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # get all the xyz files in the input directory
    xyz_files = [f for f in os.listdir(input_path) if f.endswith('.xyz')]

    for file in tqdm(xyz_files, desc="Processing molecules", total=len(xyz_files)):
        sdf_file = file.replace('.xyz', '.sdf')
        process_single_molecule(os.path.join(input_path, file), os.path.join(output_path, sdf_file), model, distance_cutoff, mode, kekulize, progress_bar=False, device=device, save_filtered=save_filtered)

def process_single_molecule(input_path, output_path, model=None, distance_cutoff=3.0, mode='3d', kekulize=False, progress_bar=True, device=None, save_filtered=None):
    
    # Read the input molecule
    # print(f'Reading molecule from {input_path}')
    mols = read_molecules(input_path)

    if output_path is None:
        if input_path.endswith('.xyz'):
            output_path = input_path.replace('.xyz', '_yuel.sdf')
        elif input_path.endswith('.sdf'):
            output_path = input_path.replace('.sdf', '_yuel.sdf')
        else:
            raise ValueError(f'Unknown file extension: {input_path}')

    # clear the output file
    open(output_path, 'w').close()

    generator = tqdm(enumerate(mols), desc="Processing molecules", total=len(mols)) if progress_bar else enumerate(mols)
    for imol, mol in generator:
        try:
            positions, one_hot, bonds = parse_molecule(mol)
            if mode != '2d':
                bonds = np.array([])
            
            # create the dataset with raw_data
            raw_data = [{
                'name': f'mol_{imol+1}',
                'positions': positions,
                'atoms': one_hot,
                'bonds': bonds,
            }]

            dataset = BondDataset(raw_data=raw_data, device=device, has_bonds=mode=='2d', progress_bar=False)
            dataloader = get_dataloader(dataset, batch_size=1, collate_fn=collate)
            data = next(iter(dataloader))

            edge_pred = model.forward(data)
            
            # Create a molecule with the predicted bonds
            mol_pred = create_molecule_from_predictions(
                positions=data['positions'],
                one_hot=data['one_hot'],
                edge_index=data['edge_index'],
                edge_pred=edge_pred,
                node_mask=data['node_mask'],
                edge_mask=data['edge_mask'],
                name=data['name'][0]
            )
        except Exception as e:
            print(f'Error processing molecule {data["name"][0]}')
            print(e)
            continue

        if save_filtered:
            append_mol_to_sdf(mol, save_filtered)
        append_mol_to_sdf(mol_pred, output_path)
                
def append_mol_to_sdf(mol, filename):
    with open(filename, 'a') as f:
        # Header block (3 lines)
        f.write(f"{mol.GetProp('_Name') if mol.HasProp('_Name') else 'Untitled'}\n")
        f.write("  RDKit          3D\n")  # Program & comment
        f.write("\n")

        # Counts line: atoms bonds [other optional fields]
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        f.write(f"{num_atoms:3d}{num_bonds:3d}  0  0  0  0  0  0  0  0999 V2000\n")

        # Atom block (x, y, z, symbol, etc.)
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            f.write(f"{pos.x:10.4f}{pos.y:10.4f}{pos.z:10.4f} {atom.GetSymbol():<3} 0  0  0  0  0  0  0  0  0  0  0  0\n")

        # Bond block (atom1, atom2, bond type)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() + 1  # SDF uses 1-based indexing
            a2 = bond.GetEndAtomIdx() + 1
            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                bond_type_idx = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                bond_type_idx = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                bond_type_idx = 3
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_type_idx = 4
            else:
                bond_type_idx = 0
                
            f.write(f"{a1:3d}{a2:3d}{bond_type_idx:3d}  0  0  0\n")

        f.write("M  END\n")

        # Properties (optional)
        # iterate over the properties and write them to the file
        for prop_name, prop_value in mol.GetPropsAsDict(True).items():
            f.write(f"> <{prop_name}>\n")
            f.write(f"{prop_value}\n\n")

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() + 1
            a2 = bond.GetEndAtomIdx() + 1
            a1, a2 = sorted([a1, a2])
            for prop_name, prop_value in bond.GetPropsAsDict(True).items():
                f.write(f"> <Bond_{a1}_{a2}_{prop_name}>\n")
                f.write(f"{prop_value}\n\n")

        # End delimiter
        f.write("$$$$\n")

def process_dataset(dataset_path, model, output_dir, batch_size=1, device=None, mode='3d', kekulize=False):
    # Load the dataset
    print(f'Loading dataset from {dataset_path}')
    # split dataset_path to the data_path and prefix
    # if there is no "/", then data_path is "."
    if '/' not in dataset_path:
        data_path = '.'
        prefix = dataset_path
    else:
        data_path, prefix = dataset_path.split('/')
    dataset = BondDataset(data_path=data_path, prefix=prefix, device=device, has_bonds=mode=='2d')
    dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each batch
    all_predictions = []
    pred_molecules = []
    true_molecules = []
    
    for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Get predictions
        edge_pred = model.forward(data)
        
        # Store predictions
        all_predictions.append({
            'edge_pred': edge_pred,
            'edge_index': data['edge_index'],
            'edge_mask': data['edge_mask'],
            'positions': data['positions'],
            'one_hot': data['one_hot'],
            'node_mask': data['node_mask'],
            'name': data['name']
        })
        
        # Create molecules with predicted bonds
        for i in range(len(data['name'])):
            mol = create_molecule_from_predictions(
                positions=data['positions'][i:i+1],
                one_hot=data['one_hot'][i:i+1],
                edge_index=data['edge_index'][i:i+1],
                edge_pred=edge_pred[i:i+1],
                node_mask=data['node_mask'][i:i+1],
                edge_mask=data['edge_mask'][i:i+1],
                name=data['name'][i]
            )
            pred_molecules.append(mol)

            mol = create_molecule_from_predictions(
                positions=data['positions'][i:i+1],
                one_hot=data['one_hot'][i:i+1],
                edge_index=data['edge_index'][i:i+1],
                edge_pred=data['bond_orders'][i:i+1],
                node_mask=data['node_mask'][i:i+1],
                edge_mask=data['edge_mask'][i:i+1],
                name=data['name'][i]
            )
            true_molecules.append(mol)
    
    # Save predictions to .pt file
    suffix = f'_{mode}'
    pt_output_path = os.path.join(output_dir, f'{prefix}_predictions{suffix}.pt')
    print(f'Saving predictions to {pt_output_path}')
    torch.save(all_predictions, pt_output_path)
    
    # Save molecules to SDF file
    sdf_output_path = os.path.join(output_dir, f'{prefix}_predictions{suffix}.sdf')
    print(f'Saving molecules to {sdf_output_path}')
    open(sdf_output_path, 'w').close()
    for mol in pred_molecules:
        try:
            append_mol_to_sdf(mol, sdf_output_path)
        except Exception as e:
            print(f'Error saving molecule {mol.GetProp("name")}')
            print(e)

    # save true molecules to sdf file
    true_sdf_output_path = os.path.join(output_dir, f'{prefix}_true{suffix}.sdf')
    print(f'Saving true molecules to {true_sdf_output_path}')
    open(true_sdf_output_path, 'w').close()
    for mol in true_molecules:
        append_mol_to_sdf(mol, true_sdf_output_path)
    
    print('Done!')

if __name__ == '__main__':
    args = parser.parse_args()
    # get current folder
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # Set random seed if provided
    if args.random_seed is not None:
        set_deterministic(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model according to models/index.yml
    model = args.model
    if model is None:
        # read the index.yml file
        with open(os.path.join(current_folder, 'models/index.yml'), 'r') as f:
            index = yaml.load(f, Loader=yaml.FullLoader)
        suffix1 = 'sanitized' if not args.kekulize else 'kekulized'
        model = os.path.join(current_folder, 'models/' + index[f'geom_{suffix1}_{args.mode}'] + '/last.ckpt')
    # print(f'Loading model from {model}')
    yuel_bond = YuelBond.load_from_checkpoint(model, map_location=device).eval().to(device)
    
    # Process single molecule or dataset
    if not args.dataset:
        if os.path.isdir(args.input):
            process_folder(
                input_path=args.input,
                output_path=args.output,
                model=yuel_bond,
                distance_cutoff=args.distance_cutoff,
                mode=args.mode,
                kekulize=args.kekulize,
                device=device,
                save_filtered=args.save_filtered
            )
        else:
            process_single_molecule(
                input_path=args.input,
                output_path=args.output,
                model=yuel_bond,
                distance_cutoff=args.distance_cutoff,
                mode=args.mode,
                kekulize=args.kekulize,
                progress_bar=True,
                device=device,
                save_filtered=args.save_filtered
            )
    else:
        process_dataset(
            dataset_path=args.input,
            output_dir=args.output,
            model=yuel_bond,
            batch_size=args.batch_size,
            mode=args.mode,
            kekulize=args.kekulize,
            device=device
        )
