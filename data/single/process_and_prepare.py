import os
import random
import torch
import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem import Descriptors, Lipinski, Recap, BRICS, FragmentOnBonds, AllChem, DataStructs
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import GetMolFrags
from Bio.PDB import PDBParser
import itertools
import pandas as pd
import pickle
import argparse
import time
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    filename='invalid_data.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 片段验证函数
def is_valid_fragment(fragment):
    """检查片段是否符合 FBDD 标准"""
    if fragment is None:
        return False
    try:
        # Chem.SanitizeMol(fragment, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.SanitizeMol(fragment)
    except Exception as e:
        logging.debug(f"Fragment sanitization failed (non-kekulized): {e}")
        return False
    if fragment.GetNumAtoms() < 5:
        return False
    mol_weight = Descriptors.MolWt(fragment)
    if mol_weight >= 300:
        return False
    logP = Descriptors.MolLogP(fragment)
    if logP > 3:
        return False
    hbd = Lipinski.NumHDonors(fragment)
    if hbd > 3:
        return False
    hba = Lipinski.NumHAcceptors(fragment)
    if hba > 3:
        return False
    psa = Descriptors.TPSA(fragment)
    if psa > 60:
        return False
    rot_bonds = Lipinski.NumRotatableBonds(fragment)
    if rot_bonds > 3:
        return False
    return True

# 移除哑原子 (*)
def remove_dummy_atom(mol):
    """从分子中移除哑原子 (*) 并返回新分子"""
    try:
        emol = Chem.EditableMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                emol.RemoveAtom(atom.GetIdx())
                break
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception as e:
        logging.debug(f"Failed to remove dummy atom: {e}")
        return None




# 切割方法 2: Recap 分解
def cut_by_recap(mol):
    fragments = []
    try:
        decomp = Recap.RecapDecompose(mol)
        leaves = decomp.GetLeaves()
        for node in leaves.values():
            fmol = node.mol
            if fmol and is_valid_fragment(fmol):
                fragments.append(fmol)
    except Exception as e:
        logging.debug(f"Recap cutting error: {e}")
    return fragments

# 切割方法 3: BRICS 分解
def cut_by_brics(mol):
    fragments = []
    try:
        breaks = BRICS.BRICSDecompose(mol)
        for smi in breaks:
            fmol = Chem.MolFromSmiles(smi)
            if fmol and is_valid_fragment(fmol):
                fragments.append(fmol)
    except Exception as e:
        logging.debug(f"BRICS cutting error: {e}")
    return fragments


# 选择最佳片段
def get_valid_fragments(mol):
    """生成有效片段，若无候选片段则返回 None，若候选片段不满足筛选条件则随机返回一个"""
    clean_mol = remove_dummy_atom(mol)
    if clean_mol is None:
        logging.warning(f"清理分子失败，返回 None: {Chem.MolToSmiles(mol)}")
        return None
    methods = [cut_by_recap, cut_by_brics]
    original_smi = Chem.MolToSmiles(clean_mol)
    try:
        original_fp = AllChem.RDKFingerprint(clean_mol)
        original_atoms = clean_mol.GetNumAtoms()
    except Exception as e:
        logging.warning(f"生成原始骨架指纹失败: {e}, SMILES: {original_smi}")
        return None

    candidate_fragments = []
    for method in methods:
        fragments = method(clean_mol)
        for frag_mol in fragments:
            frag_smi = Chem.MolToSmiles(frag_mol)
            frag_fp = AllChem.RDKFingerprint(frag_mol)
            if frag_smi != original_smi :
                try:
                    # Chem.SanitizeMol(frag_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    candidate_fragments.append(frag_mol)
                except Exception as e:
                    logging.debug(f"无效片段已跳过: {frag_smi}, 错误: {e}")
                    continue

    if len(candidate_fragments) == 0:
        logging.warning(f"未找到任何候选片段，切割失败，返回 None: {original_smi}")
        return None
    elif len(candidate_fragments) == 1:
        return Chem.MolToSmiles(candidate_fragments[0])
    else:
        valid_candidates = []
        for frag_mol in candidate_fragments:
            try:
                frag_fp = AllChem.RDKFingerprint(frag_mol)
                similarity = DataStructs.TanimotoSimilarity(original_fp, frag_fp)
                if 0.2 <= similarity <= 0.8:
                    valid_candidates.append((frag_mol, similarity))
            except Exception:
                continue
        if valid_candidates:
            best_fragment, _ = max(valid_candidates, key=lambda x: x[1])
            return Chem.MolToSmiles(best_fragment)
        else:
            logging.debug(f"多个候选片段均不满足筛选条件，随机选择一个: {original_smi}")
            selected_fragment = random.choice(candidate_fragments)
            return Chem.MolToSmiles(selected_fragment)

# 处理输入文件，只保留片段信息
def process_input_file(input_file, output_file):
    """处理输入 SMILES 文件，只生成片段信息"""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_data = []
    with tqdm(total=len(lines), desc="处理输入 SMILES") as pbar:
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                logging.warning(f"无效行格式: {line.strip()}")
                pbar.update(1)
                continue
            scaf_smi, rgroup_smi, mol_smi = parts
            scaf_mol = Chem.MolFromSmiles(scaf_smi)
            if scaf_mol is None:
                logging.warning(f"无效骨架 SMILES: {scaf_smi}")
                pbar.update(1)
                continue
            fragment_smi = get_valid_fragments(scaf_mol)
            if fragment_smi is None:
                logging.warning(f"切割失败，无有效片段，跳过: {scaf_smi}")
                pbar.update(1)
                continue
            frag_mol = Chem.MolFromSmiles(fragment_smi)
            if frag_mol is None or fragment_smi == scaf_smi:
                logging.warning(f"片段无效或等于骨架，跳过: {scaf_smi}")
                pbar.update(1)
                continue
            processed_data.append((scaf_smi, rgroup_smi, mol_smi, fragment_smi))
            pbar.update(1)

    with open(output_file, 'w') as f:
        for data in processed_data:
            f.write('\t'.join(str(x) for x in data) + '\n')
    return output_file

# 口袋提取
def get_pocket(mol, pdb_path, ligand_filename=None, scaf_smi=None, rgroup_smi=None):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(resid)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    if atom_coords.size == 0:
        logging.warning(
            f"No atoms found in protein structure from {pdb_path}, skipping... "
            f"ligand_filename: {ligand_filename}, "
            f"protein_filename: {pdb_path}, "
            f"scaf_smi: {scaf_smi}, "
            f"rgroup_smi: {rgroup_smi}"
        )
        return None

    mol_atom_coords = mol.GetConformer().GetPositions()
    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(1) <= 6)[0]])

    pocket_coords_full = []
    pocket_types_full = []
    pocket_coords_bb = []
    pocket_types_bb = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        if resid not in contact_residues:
            continue
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_type = atom.element.upper()
            atom_coord = atom.get_coord()
            pocket_coords_full.append(atom_coord.tolist())
            pocket_types_full.append(atom_type)
            if atom_name in {'N', 'CA', 'C', 'O'}:
                pocket_coords_bb.append(atom_coord.tolist())
                pocket_types_bb.append(atom_type)

    if not pocket_coords_full:
        logging.warning(
            f"No pocket coordinates found for {pdb_path}, skipping... "
            f"ligand_filename: {ligand_filename}, "
            f"protein_filename: {pdb_path}, "
            f"scaf_smi: {scaf_smi}, "
            f"rgroup_smi: {rgroup_smi}"
        )
        return None

    return {
        'full_coord': pocket_coords_full,
        'full_types': pocket_types_full,
        'bb_coord': pocket_coords_bb,
        'bb_types': pocket_types_bb,
    }

# 辅助函数
def get_exits(mol):
    exits = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            exits.append(atom)
    return exits

def set_anchor_flags(mol, anchor_idx):
    for atom in mol.GetAtoms():
        atom.SetProp('_Anchor', '1' if atom.GetIdx() == anchor_idx else '0')

def update_scaffold(scaf):
    star_symbol = get_exits(scaf)
    if len(star_symbol) > 1:
        raise Exception('Found more than one exit in scaffold')
    star_symbol = star_symbol[0]
    bonds = star_symbol.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    exit_idx = star_symbol.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(scaf, anchor_idx)
    escaffold = Chem.EditableMol(scaf)
    escaffold.RemoveBond(source_idx, target_idx)
    escaffold.RemoveAtom(exit_idx)
    return escaffold.GetMol()

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)
    if len(matches) < 1:
        raise Exception('Could not find scaffold or rgroup matches')
    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        scaf_coords = mol_coords[np.array(match)]
        scaf_conformer = create_conformer(scaf_coords)
        match2conf[match] = scaf_conformer
    return match2conf

def find_non_intersecting_matches(matches1, matches2):
    triplets = list(itertools.product(matches1, matches2))
    non_intersecting_matches = set()
    for m1, m2 in triplets:
        if not set(m1) & set(m2):
            non_intersecting_matches.add((m1, m2))
    return list(non_intersecting_matches)

def find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol):
    matches_with_rgroup_in_the_middle = []
    for m1, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbors.add(neighbor.GetIdx())
        conn1 = set(m1) & neighbors
        if len(conn1) == 1:
            matches_with_rgroup_in_the_middle.append((m1, lm))
    return matches_with_rgroup_in_the_middle

def find_correct_matches(matches_scaf, matches_rgroup, mol):
    non_intersecting_matches = find_non_intersecting_matches(matches_scaf, matches_rgroup)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches
    return find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol)

def prepare_scaffold_and_rgroup(scaf_smi, rgroup_smi, mol):
    if isinstance(rgroup_smi, list):
        rgroup_smi = rgroup_smi[0]
    scaf = Chem.MolFromSmiles(scaf_smi)
    rgroup = Chem.MolFromSmiles(rgroup_smi)

    newscaf = update_scaffold(scaf)
    newrgroup = update_scaffold(rgroup)

    match2conf_scaf = transfer_conformers(newscaf, mol)
    match2conf_rgroup = transfer_conformers(newrgroup, mol)

    correct_matches = find_correct_matches(
        match2conf_scaf.keys(),
        match2conf_rgroup.keys(),
        mol,
    )

    if len(correct_matches) > 2:
        raise Exception('Found more than two scaffold matches')

    conf_scaf = match2conf_scaf[correct_matches[0][0]]
    conf_rgroup = match2conf_rgroup[correct_matches[0][1]]
    newscaf.AddConformer(conf_scaf)
    newrgroup.AddConformer(conf_rgroup)

    return newscaf, newrgroup

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())
    return anchors_idx




def remove_all_dummy_atoms(mol_with_dummies):
    """
    从分子中移除所有类型的哑原子 (原子序数为0的原子)。
    返回一个新的、不含哑原子的RDKit Mol对象，如果失败则返回None。
    """
    if mol_with_dummies is None:
        return None

    rw_mol = Chem.RWMol(mol_with_dummies)
    atoms_to_remove_indices = []
    has_dummy_atoms = False
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # 哑原子的原子序数为0
            atoms_to_remove_indices.append(atom.GetIdx())
            has_dummy_atoms = True

    if not has_dummy_atoms:  # 如果没有哑原子
        try:
            # 确保原始mol也被净化 (如果它之前没有被完全净化)
            # 但如果 mol_with_dummies 来自 Chem.MolFromSmiles, 它已经被净化过了
            # Chem.SanitizeMol(rw_mol)
            return rw_mol.GetMol()
        except Exception as e:
            logging.debug(f"Error sanitizing molecule without dummies: {e}")
            return None

    for idx in sorted(atoms_to_remove_indices, reverse=True):
        rw_mol.RemoveAtom(idx)

    try:
        final_mol = rw_mol.GetMol()
        if final_mol and final_mol.GetNumAtoms() > 0:  # 确保移除后不是空分子
            Chem.SanitizeMol(final_mol)  # 净化清理后的分子
            return final_mol
        else:
            logging.debug(
                f"Molecule became empty after removing dummy atoms from: {Chem.MolToSmiles(mol_with_dummies) if mol_with_dummies else 'None'}")
            return None
    except Exception as e:
        logging.debug(
            f"Failed to sanitize molecule after removing all dummy atoms from {Chem.MolToSmiles(mol_with_dummies) if mol_with_dummies else 'None'}: {e}")
        return None
def process_sdf(scaf_dataset):
    """处理 SDF 数据并生成分子、骨架、片段等"""
    molecules, scaffolds, rgroups, pockets, fragments, out_table = [], [], [], [], [], []
    uuid = 0
    total = len(scaf_dataset['scaf_smi'])

    with tqdm(total=total, desc="处理 SDF 数据") as pbar:
        for i in range(total):
            ligand_filename = os.path.join('/home/cht/DiffDec-master/data/crossdocked_pocket10',
                                           scaf_dataset['ligand_filename'][i])
            protein_filename = os.path.join('/home/cht/DiffDec-master/data/crossdocked_pocket10',
                                            scaf_dataset['protein_filename'][i])
            scaf_smi = scaf_dataset['scaf_smi'][i]
            rgroup_smi = scaf_dataset['rgroup_smi'][i]
            fragment_smi_from_input = scaf_dataset['fragment_smi'][i]  # 重命名以示区分

            # 1. 加载和清理配体 SDF
            try:
                mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
                if mol is None:
                    logging.warning(f"加载 SDF 失败: {ligand_filename}")
                    pbar.update(1)
                    continue
                Chem.SanitizeMol(mol)  # 默认会进行Kekulization
            except Exception as e:
                logging.warning(f"清理 {ligand_filename} 失败: {e}")
                pbar.update(1)
                continue

            mol_name = mol.GetProp('_Name')
            mol_smi_full = Chem.MolToSmiles(mol)  # 获取完整分子的SMILES

            # 2. 提取口袋
            pocket = get_pocket(mol, protein_filename, ligand_filename, scaf_smi, rgroup_smi)
            if pocket is None:
                # logging.warning 应该在 get_pocket 内部，这里只更新 pbar 和 continue
                pbar.update(1)
                continue

            # 3. 准备骨架和R基团
            try:
                scaffold, rgroup = prepare_scaffold_and_rgroup(scaf_smi, rgroup_smi, mol)
            except Exception as e:
                logging.warning(f"准备 {mol_smi_full} 的骨架/侧链失败: {e}")
                pbar.update(1)
                continue

            # 4. 处理片段 (核心修改区域)
            processed_fragment_mol = None  # 用于存储最终要使用的片段对象

            fragment_mol_with_dummies = Chem.MolFromSmiles(fragment_smi_from_input)
            if fragment_mol_with_dummies:
                # 首先尝试净化带哑原子的片段，确保其本身是有效的化学结构
                try:
                    Chem.SanitizeMol(fragment_mol_with_dummies)
                except Exception as e:
                    logging.warning(f"净化带哑原子的片段 {fragment_smi_from_input} 失败: {e}")
                    fragment_mol_with_dummies = None  # 标记为无效

            if fragment_mol_with_dummies:
                fragment_mol_clean = remove_all_dummy_atoms(fragment_mol_with_dummies)

                if fragment_mol_clean and fragment_mol_clean.GetNumAtoms() > 0:
                    try:
                        # 使用清理后的、不含哑原子的片段进行子结构匹配
                        matches = mol.GetSubstructMatches(fragment_mol_clean, useChirality=True)  # 考虑手性匹配
                        if matches:
                            match = matches[0]  # 取第一个匹配
                            fragment_coords = mol.GetConformer().GetPositions()[list(match)]

                            # 将构象赋予清理后的片段
                            fragment_mol_clean.AddConformer(create_conformer(fragment_coords))
                            processed_fragment_mol = fragment_mol_clean  # 这个是最终要保存的片段
                        else:
                            # 尝试不带手性的匹配作为后备 (如果手性是问题)
                            matches_no_chirality = mol.GetSubstructMatches(fragment_mol_clean, useChirality=False)
                            if matches_no_chirality:
                                match = matches_no_chirality[0]
                                fragment_coords = mol.GetConformer().GetPositions()[list(match)]
                                fragment_mol_clean.AddConformer(create_conformer(fragment_coords))
                                processed_fragment_mol = fragment_mol_clean
                                logging.debug(
                                    f"片段(清理后)通过非手性匹配成功: {Chem.MolToSmiles(fragment_mol_clean)} in mol {mol_smi_full}")
                            else:
                                logging.warning(
                                    f"片段(清理后)无子结构匹配 (手性和非手性均失败): {Chem.MolToSmiles(fragment_mol_clean)} in mol {mol_smi_full}")
                                # processed_fragment_mol 保持为 None
                    except Exception as e:
                        logging.warning(
                            f"处理清理后的片段 {Chem.MolToSmiles(fragment_mol_clean)} (匹配或构象)时出错: {e}")
                        # processed_fragment_mol 保持为 None
                else:
                    logging.warning(f"从 {fragment_smi_from_input} 清理哑原子后片段无效或为空。")
                    # processed_fragment_mol 保持为 None
            else:
                logging.warning(f"创建或净化带哑原子的片段分子失败 (来自SMILES): {fragment_smi_from_input}")
                # processed_fragment_mol 保持为 None

            if processed_fragment_mol is None:  # 如果上述任何一步导致 fragment_mol 变成 None
                pbar.update(1)
                continue  # 跳过这条数据

            # 5. 如果所有步骤都成功，才将数据添加到列表中
            anchors_idx = get_anchors_idx(scaffold)  # 确保 scaffold 有效
            if not anchors_idx:  # 如果骨架没有锚点 (不应该发生如果prepare_scaffold_and_rgroup正常)
                logging.warning(f"骨架 {Chem.MolToSmiles(scaffold) if scaffold else 'None'} 无锚点，跳过 {mol_smi_full}")
                pbar.update(1)
                continue

            molecules.append(mol)
            scaffolds.append(scaffold)
            rgroups.append(rgroup)
            pockets.append(pocket)
            fragments.append(processed_fragment_mol)  # 添加处理后的片段
            out_table.append({
                'uuid': uuid,
                'molecule_name': mol_name,
                'molecule': mol_smi_full,  # 使用 mol_smi_full
                'scaffold': Chem.MolToSmiles(scaffold),
                'rgroups': Chem.MolToSmiles(rgroup),
                'anchor': anchors_idx[0],
                'pocket_full_size': len(pocket['full_types']),
                'pocket_bb_size': len(pocket['bb_types']),
                'molecule_size': mol.GetNumAtoms(),
                'scaffold_size': scaffold.GetNumAtoms(),
                'rgroup_size': rgroup.GetNumAtoms(),
                'protein_filename': protein_filename,
                'scaffold_fragment': Chem.MolToSmiles(processed_fragment_mol)  # 保存清理后的片段SMILES
                # 或者如果你想保留原始带哑原子的SMILES记录: fragment_smi_from_input
            })
            uuid += 1
            pbar.update(1)

    return molecules, scaffolds, rgroups, pockets, fragments, pd.DataFrame(out_table)
# 准备函数
def prepare(sliced_file, mode):
    """准备骨架数据集，仅包含片段 SMILES"""
    decomp_dict = {}
    with open(sliced_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 4:  # 现在只有四列
                continue
            scaf_smi, rgroup_smi, mol_smi, fragment_smi = parts
            rgroups_list = rgroup_smi.split('|')
            if mol_smi not in decomp_dict:
                decomp_dict[mol_smi] = [(scaf_smi, rgroups_list, fragment_smi)]
            else:
                decomp_dict[mol_smi].append((scaf_smi, rgroups_list, fragment_smi))

    scaf_dict = {
        'ligand_filename': [],
        'protein_filename': [],
        'scaf_smi': [],
        'rgroup_smi': [],
        'fragment_smi': []
    }
    split_by_name = torch.load('/home/cht/DiffDec-master/data/split_by_name.pt')
    path_prefix = '/home/cht/DiffDec-master/data/crossdocked_pocket10/'
    with tqdm(total=len(split_by_name[mode]), desc=f"Preparing {mode} data") as pbar:
        for i in range(len(split_by_name[mode])):
            ligand_filename = path_prefix + split_by_name[mode][i][1]
            mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            if smi not in decomp_dict:
                pbar.update(1)
                continue
            for scaf_smi, rgroup_list, fragment_smi in decomp_dict[smi]:
                scaf_dict['ligand_filename'].append(split_by_name[mode][i][1])
                scaf_dict['protein_filename'].append(split_by_name[mode][i][0])
                scaf_dict['scaf_smi'].append(scaf_smi)
                scaf_dict['rgroup_smi'].append(rgroup_list)
                scaf_dict['fragment_smi'].append(fragment_smi)
            pbar.update(1)
    return scaf_dict

# 主函数
def main(scaf_dataset, mode):
    """生成 SDF 文件和 CSV 表格，不包含 remain 相关内容"""
    out_mol_path = f'./crossdock_{mode}_mol.sdf'
    out_scaf_path = f'./crossdock_{mode}_scaf.sdf'
    out_rgroup_path = f'./crossdock_{mode}_rgroup.sdf'
    out_pockets_path = f'./crossdock_{mode}_pockets.pkl'
    out_table_path = f'./crossdock_{mode}_table.csv'
    out_fragment_path = f'./crossdock_{mode}_fragment.sdf'

    molecules, scaffolds, rgroups, pockets, fragments, out_table = process_sdf(scaf_dataset)

    with Chem.SDWriter(out_mol_path) as writer:
        for mol in molecules:
            writer.write(mol)
    with Chem.SDWriter(out_scaf_path) as writer:
        writer.SetKekulize(False)
        for scaf in scaffolds:
            writer.write(scaf)
    with Chem.SDWriter(out_rgroup_path) as writer:
        writer.SetKekulize(False)
        for rgroup in rgroups:
            writer.write(rgroup)
    with Chem.SDWriter(out_fragment_path) as writer:
        writer.SetKekulize(False)
        for fragment in fragments:
            writer.write(fragment)
    with open(out_pockets_path, 'wb') as f:
        pickle.dump(pockets, f)
    out_table.to_csv(out_table_path, index=False)

    # 验证数量
    scaf_count = len(scaffolds)
    frag_count = len(fragments)
    print(f"{mode} 模式: scaf.sdf: {scaf_count}, fragment.sdf: {frag_count}")
    if scaf_count == frag_count:
        print(f"{mode} 模式: scaf.sdf 和 fragment.sdf 数量一致。")
    else:
        print(f"{mode} 模式: 警告: scaf.sdf 和 fragment.sdf 数量不一致！")

if __name__ == '__main__':
    # 处理输入文件
    train_sliced_file = '../train_smi_out.smi'  # libinvent处理后的文件路径
    test_sliced_file = '../test_smi_out.smi'
    processed_train_file = '../train_smi_out_processed.smi'  # 最终数据集处理的结果路径
    processed_test_file = '../test_smi_out_processed.smi'

    print("Processing train_smi_out.smi...")
    process_input_file(train_sliced_file, processed_train_file)
    print("Processing test_smi_out.smi...")
    process_input_file(test_sliced_file, processed_test_file)

    # 准备并处理数据
    print("Preparing train data...")
    scaf_train = prepare(processed_train_file, 'train')
    print("Preparing test data...")
    scaf_test = prepare(processed_test_file, 'test')

    print("Running main function for train data...")
    main(scaf_train, 'train')
    print("Running main function for test data...")
    main(scaf_test, 'test')
    print("Processing complete.")
