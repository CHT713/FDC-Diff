import torch
from rdkit import Chem

split_by_name = torch.load('split_by_name.pt')
path_prefix = '/home/cht/DiffDec-master/data/crossdocked_pocket10/'
f = open('train_smi1.smi', 'w')
for i in range(len(split_by_name['train'])):
    ligand_filename = path_prefix + split_by_name['train'][i][1]
    try:
        mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
    except Exception as e:
        print(f"读取文件 {ligand_filename} 时出错: {e}")

    mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    f.write(smi + '\n')
f.close()
f = open('test_smi.smi1', 'w')
for i in range(len(split_by_name['test'])):
    ligand_filename = path_prefix + split_by_name['test'][i][1]
    mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    f.write(smi + '\n')
f.close()
