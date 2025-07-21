import os
import pickle
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

import torch
from scipy.stats import truncnorm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import const
from functools import lru_cache

# ----------------------------
# 1. 分子读取与基本编码函数
# ----------------------------
def read_sdf(sdf_path):
    """读取SDF文件并生成分子对象"""
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for mol in suppl:
        if mol is not None:
            yield mol

def get_one_hot(atom_symbol, atoms_dict):
    """生成原子的one-hot编码"""
    one_hot = np.zeros(len(atoms_dict), dtype=np.float32)
    one_hot[atoms_dict[atom_symbol]] = 1.0
    return one_hot

# ----------------------------
# 2. 分子预处理与清洗函数
# ----------------------------
def sanitize_molecule(mol):
    """分子预处理：仅在需要时使用"""
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol.ToBinary())
        m = Chem.RemoveAllHs(m)
        Chem.SanitizeMol(m)
        for atom in m.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return m
    except Exception as e:
        print(f"Sanitize 失败: {e}")
        return None

# ---------------------------------
# 3. 片段与骨架分子数据分离函数
# ---------------------------------
import numpy as np
from rdkit import Chem

def parse_scaffold_with_fragment(scaffold_mol, fragment_mol):
    """
    在骨架分子中找到 fragment_mol 对应的片段，并拆分出:
      - 片段 (frag) 的原子坐标、特征、全局索引
      - 剩余 (rem) 的原子坐标、特征、全局索引
      - 跨片段-剩余的连接信息

    新增:
      - frag_attach_rem_mask: 长度=片段原子数, 如果该片段原子与剩余部分相连则=1, 否则=0
      - rem_attach_frag_mask: 长度=剩余部分原子数, 如果该剩余原子与片段相连则=1, 否则=0
    """

    try:
        # 1. 子结构匹配
        matches = scaffold_mol.GetSubstructMatches(fragment_mol)
        if not matches:
            print("片段未在骨架分子中匹配")
            return None
        frag_indices = list(matches[0])  # 只取第一个匹配


        # 2. 基础信息提取
        conf = scaffold_mol.GetConformer()
        scaf_pos = conf.GetPositions()  # shape=(N,3)
        scaf_one_hot = [get_one_hot(atom.GetSymbol(), const.ATOM2IDX) for atom in scaffold_mol.GetAtoms()]
        scaf_charges = [const.CHARGES[atom.GetSymbol()] for atom in scaffold_mol.GetAtoms()]
        scaf_degrees = np.array([atom.GetDegree() for atom in scaffold_mol.GetAtoms()], dtype=np.float32)

        max_deg = scaf_degrees.max() if len(scaf_degrees) > 0 else 0
        scaf_degrees_norm = scaf_degrees / max_deg if max_deg > 0 else scaf_degrees

        # 3. 拆分片段与剩余部分
        frag_pos = scaf_pos[frag_indices]
        frag_hot = np.array([scaf_one_hot[i] for i in frag_indices])
        frag_chg = np.array([scaf_charges[i] for i in frag_indices])
        frag_deg = scaf_degrees_norm[frag_indices]

        N = len(scaf_pos)
        remain_mask = np.ones(N, dtype=bool)
        remain_mask[frag_indices] = False
        rem_indices = np.where(remain_mask)[0]

        rem_pos = scaf_pos[rem_indices]
        rem_hot = np.array([scaf_one_hot[i] for i in rem_indices])
        rem_chg = np.array([scaf_charges[i] for i in rem_indices])
        rem_deg = scaf_degrees_norm[rem_indices]

        # 4. 检测片段<-->剩余部分的跨部分键
        #    (a) 建立局部索引映射, 全局->片段/剩余的局部 index
        frag_idx_map = {g_idx: i for i, g_idx in enumerate(frag_indices)}
        rem_idx_map = {g_idx: i for i, g_idx in enumerate(rem_indices)}

        #    (b) 记录所有跨片段-剩余的键
        attachment_bonds = []
        for bond in scaffold_mol.GetBonds():
            bgn, ed = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (bgn in frag_idx_map and ed in rem_idx_map):
                attachment_bonds.append({
                    'frag_atom_global': bgn,
                    'frag_atom_local': frag_idx_map[bgn],
                    'rem_atom_global': ed,
                    'rem_atom_local': rem_idx_map[ed],
                    'bond_type': encode_bond_type(bond.GetBondType())
                })
            elif (ed in frag_idx_map and bgn in rem_idx_map):
                attachment_bonds.append({
                    'frag_atom_global': ed,
                    'frag_atom_local': frag_idx_map[ed],
                    'rem_atom_global': bgn,
                    'rem_atom_local': rem_idx_map[bgn],
                    'bond_type': encode_bond_type(bond.GetBondType())
                })

        # 5. 根据 attachment_bonds 生成片段与剩余部分各自的掩码
        frag_attach_rem_mask = np.zeros(len(frag_indices), dtype=np.float32)
        rem_attach_frag_mask = np.zeros(len(rem_indices), dtype=np.float32)
        for bond_info in attachment_bonds:
            f_local = bond_info['frag_atom_local']
            r_local = bond_info['rem_atom_local']
            frag_attach_rem_mask[f_local] = 1.0
            rem_attach_frag_mask[r_local] = 1.0

        # 6. 生成骨架整体的 edge_index / bond_types (如你所需的图结构)
        edges = []
        bond_types = []
        for bond in scaffold_mol.GetBonds():
            btype = encode_bond_type(bond.GetBondType())
            bgn, ed = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # 根据需要, 双向边
            edges.append((bgn, ed))
            bond_types.append(btype)
            edges.append((ed, bgn))
            bond_types.append(btype)

        edge_index = np.array(edges, dtype=np.int32).T  # shape=(2,E)
        bond_type_arr = np.array(bond_types, dtype=np.int32)  # shape=(E,)

        # 7. 适配 all_positions 的索引
        # all_positions = np.concatenate([frag_pos, rem_pos, rgroup_pos, pocket_pos])
        # 构建从骨架全局索引到 all_positions 索引的映射
        scaffold_to_all_positions = {}
        for i, g_idx in enumerate(frag_indices):
            scaffold_to_all_positions[g_idx] = i  # 片段原子：直接映射到 0 到 len(frag_pos)-1
        for i, g_idx in enumerate(rem_indices):
            scaffold_to_all_positions[g_idx] = len(frag_pos) + i  # 剩余部分原子：偏移 len(frag_pos)

        # 转换 edge_index 到 all_positions 的索引
        all_edge_index = np.array([
            [scaffold_to_all_positions[b], scaffold_to_all_positions[e]]
            for b, e in edge_index.T
        ]).T


        # 7. 返回一个 dict
        return  frag_pos,frag_hot,frag_chg,frag_deg,frag_indices,frag_attach_rem_mask,rem_pos, rem_hot,rem_chg,rem_deg,rem_indices,rem_attach_frag_mask,remain_mask,attachment_bonds, edge_index,all_edge_index,bond_type_arr


    except Exception as e:
        print(f"[parse_scaffold_with_fragment] 出错: {e}")
        return None

import numpy as np

def map_scaf_attach_rgroup_to_frag_rem(
    scaf_attach_rgroup_mask,
    frag_global_idx,
    rem_global_idx
):
    """
    将骨架层面的 scaf_attach_rgroup_mask (长度 = scaffold_mol.GetNumAtoms())，
    映射到 frag 和 rem 的局部索引上。

    参数：
      scaf_attach_rgroup_mask : np.ndarray
        shape=(N_scaf,), 值为1表示该骨架原子与R基团相连，0表示未连接
      frag_global_idx : List[int] or np.ndarray
        frag部分的“全局原子索引”（在 scaffold_mol 中）
      rem_global_idx : List[int] or np.ndarray
        rem部分的“全局原子索引”（在 scaffold_mol 中）

    返回：
      frag_attach_rgroup_mask : np.ndarray
        shape=(len(frag_global_idx),), frag里哪些原子与R基团相连
      rem_attach_rgroup_mask : np.ndarray
        shape=(len(rem_global_idx),), rem里哪些原子与R基团相连
    """

    # 1. 初始化两个掩码，全为0
    frag_attach_rgroup_mask = np.zeros(len(frag_global_idx), dtype=np.float32)
    rem_attach_rgroup_mask = np.zeros(len(rem_global_idx), dtype=np.float32)

    # 2. frag部分：遍历frag局部原子索引 -> 其在scaffold_mol的全局索引
    for local_i, scaf_idx in enumerate(frag_global_idx):
        # 如果在骨架掩码中该原子是1，说明它和R基团相连
        if scaf_attach_rgroup_mask[scaf_idx] == 1.0:
            frag_attach_rgroup_mask[local_i] = 1.0

    # 3. rem部分：同理
    for local_j, scaf_idx in enumerate(rem_global_idx):
        if scaf_attach_rgroup_mask[scaf_idx] == 1.0:
            rem_attach_rgroup_mask[local_j] = 1.0

    return frag_attach_rgroup_mask, rem_attach_rgroup_mask


def encode_bond_type(rdkit_bond_type):
    """将 RDKit 的键类型转成整型编码, 单键=1, 双键=2, 三键=3, 芳香=4, 其余=0"""
    from rdkit import Chem
    if rdkit_bond_type == Chem.rdchem.BondType.SINGLE:
        return 1
    elif rdkit_bond_type == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif rdkit_bond_type == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif rdkit_bond_type == Chem.rdchem.BondType.AROMATIC:
        return 4
    else:
        return 0


# ----------------------------
# 4. 骨架匹配及最佳匹配选择
# ----------------------------
def validate_scaffold(full_mol, scaffold_mol):
    """改进后的骨架匹配验证，确保 scaffold_mol 完全存在于 full_mol 中"""
    full_mol = sanitize_molecule(full_mol)
    if not full_mol or not scaffold_mol:
        print("分子 Sanitize 失败，跳过匹配")
        return None
    try:
        matches = full_mol.GetSubstructMatches(scaffold_mol, useChirality=False)
        if not matches:
            print("骨架未在完整分子中匹配")
            return None
        best_match = matches[0]
        return best_match
    except Exception as e:
        print(f"骨架匹配失败: {str(e)}")
        return None

# -------------------------------------
# 5. 连接点检测与拓扑特征提取
# -------------------------------------
def find_attachment_and_topology(full_mol, scaffold_mol, rgroup_mol):
    """
    改进的连接点检测:
      1. 在 full_mol 中找到 scaffold_mol 的匹配 (best_scaf_match)
      2. rgroup_mol 与 scaffold_mol 区分后提取 R 基团位置
      3. 新增返回:
         - scaf_attach_rgroup_mask: 长度 = scaffold_mol原子数, 1表示该骨架原子与rgroup相连
         - rgroup_attach_scaf_mask: 长度 = rgroup_mol原子数, 1表示该rgroup原子与骨架相连
    """
    try:
        best_scaf_match = validate_scaffold(full_mol, scaffold_mol)
        if best_scaf_match is None:
            print("骨架未在完整分子中匹配")
            return None

        scaf_match_set = set(best_scaf_match)
        all_atoms = set(range(full_mol.GetNumAtoms()))
        rgroup_match = list(all_atoms - scaf_match_set)
        if len(rgroup_match) < 1:
            print("未检测到有效 R 基团")
            return None

        # -- 1. 骨架与R基团之间的连接点 --
        attachment_scaf = []
        attachment_rgroup = []
        for bond in full_mol.GetBonds():
            bgn, ed = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (bgn in scaf_match_set and ed in rgroup_match) or \
                    (ed in scaf_match_set and bgn in rgroup_match):
                if bgn in scaf_match_set:
                    attachment_scaf.append(bgn)
                    attachment_rgroup.append(ed)
                else:
                    attachment_scaf.append(ed)
                    attachment_rgroup.append(bgn)

        attachment_scaf = list(set(attachment_scaf))
        attachment_rgroup = list(set(attachment_rgroup))

        # -- 2. 提取R基团坐标、特征
        full_conf = full_mol.GetConformer()
        rgroup_pos = np.array([full_conf.GetAtomPosition(i) for i in rgroup_match])
        rgroup_hot = [get_one_hot(full_mol.GetAtomWithIdx(i).GetSymbol(), const.ATOM2IDX) for i in rgroup_match]
        rgroup_charges = [const.CHARGES[full_mol.GetAtomWithIdx(i).GetSymbol()] for i in rgroup_match]

        # 先置0
        rgroup_attach_scaf_mask = np.zeros(len(rgroup_pos), dtype=np.float32)
        for idx in attachment_rgroup:
            if idx in rgroup_match:
                local_idx = rgroup_match.index(idx)
                rgroup_attach_scaf_mask[local_idx] = 1.0

        # -- 3. 提取骨架坐标, 构建scaf_attach_rgroup_mask
        scaf_conf = scaffold_mol.GetConformer()
        scaf_pos = np.array([scaf_conf.GetAtomPosition(i) for i in range(scaffold_mol.GetNumAtoms())])

        scaf_attach_rgroup_mask = np.zeros(len(scaf_pos), dtype=np.float32)

        # 将full_mol的全局原子索引 => scaffold_mol的局部索引
        # best_scaf_match[i_scaf] = full_idx
        # full_to_scaf_map[full_idx] = i_scaf
        full_to_scaf_map = {full_idx: i_scaf for i_scaf, full_idx in enumerate(best_scaf_match)}

        for full_idx in attachment_scaf:
            if full_idx in full_to_scaf_map:
                scaf_idx = full_to_scaf_map[full_idx]
                if 0 <= scaf_idx < len(scaf_pos):
                    scaf_attach_rgroup_mask[scaf_idx] = 1.0

        # -- 4. 计算几何特征(可保持你原先逻辑)
        #    例如:


        # -- 5. 也可以在这里做full_mol的edge_index提取(如需要)
        edges = []
        bond_types = []
        for bond in full_mol.GetBonds():
            btype = encode_bond_type(bond.GetBondType())
            bgn, ed = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((bgn, ed))
            bond_types.append(btype)
            edges.append((ed, bgn))
            bond_types.append(btype)

        edge_index = np.array(edges, dtype=np.int32).T
        bond_type_arr = np.array(bond_types, dtype=np.int32)

        return   rgroup_pos,rgroup_hot,rgroup_charges,rgroup_attach_scaf_mask, scaf_pos,scaf_attach_rgroup_mask, edge_index, bond_type_arr

    except Exception as e:
        print(f"处理连接点和拓扑特征失败: {str(e)}")
        return None

@lru_cache(maxsize=1000)
def get_topo_features(mol):
    """计算原子级拓扑特征"""
    try:
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj_matrix)  # 使用 from_numpy_array 替代 from_numpy_matrix（已弃用）
        betweenness = nx.betweenness_centrality(G) # 介数中心性表示一个节点在其他节点之间的最短路径中的出现频率，节点越常出现在最短路径上，其介数中心性越高，通常用于衡量节点在图中的控制力。
        closeness = nx.closeness_centrality(G) # 紧密度中心性是节点到其他所有节点的平均距离，距离越短，节点的紧密度中心性越高，反映了节点在图中的接近程度
        return {
            'betweenness': np.array(list(betweenness.values())),  # 修正语法错误：缺失括号
            'closeness': np.array(list(closeness.values()))
        }
    except Exception:
        return None

def parse_rgroup_with_fake_atoms(rgroup_pos, rgroup_hot, rgroup_charges, rgroup_attachment, fake_pos, max_atoms=10):
    """填充 R 基团到固定原子数"""
    num_atoms = len(rgroup_pos)
    num_fake = max_atoms - num_atoms
    if num_fake > 0:
        # 假原子的位置
        fake_pos_list = np.array([fake_pos] * num_fake)
        # 假原子的独热编码，使用 '#' 表示假原子
        fake_hot = np.array([get_one_hot('#', const.ATOM2IDX)] * num_fake)
        # 假原子的电荷，假设为 0
        fake_charges = np.array([const.CHARGES['#']] * num_fake)
        # 假原子的连接点标记，填充为 0
        fake_attachment = np.zeros(num_fake, dtype=np.float32)
        # 填充
        rgroup_pos = np.concatenate([rgroup_pos, fake_pos_list])
        rgroup_hot = np.concatenate([rgroup_hot, fake_hot])
        rgroup_charges = np.concatenate([rgroup_charges, fake_charges])
        rgroup_attachment = np.concatenate([rgroup_attachment, fake_attachment])
    return rgroup_pos, rgroup_hot, rgroup_charges, rgroup_attachment

def get_geometric_features(positions, indices):
    """计算指定原子的几何特征"""
    if not indices:
        return np.empty((0, 9), dtype=np.float32)
    indexed_positions = positions[indices]
    diffs = indexed_positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    closest_indices = np.argsort(dists, axis=1)[:, 1:4]
    row_indices = np.arange(len(indices))[:, None]
    closest_dists = dists[row_indices, closest_indices]
    return closest_dists.reshape(len(indices), -1)


def preprocess_bond_info(mol):
    """从分子对象中提取键和三元组信息"""
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    edges = np.array(np.where(adj_matrix)).T  # (num_edges, 2)

    # 生成连续三元组 (A-B-C结构)
    triplets = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        # 查找与b相连的其他原子
        for other_bond in mol.GetBonds():
            if other_bond.GetBeginAtomIdx() == b:
                c = other_bond.GetEndAtomIdx()
                if c != a:
                    triplets.append([a, b, c])
            elif other_bond.GetEndAtomIdx() == b:
                c = other_bond.GetBeginAtomIdx()
                if c != a:
                    triplets.append([c, b, a])
    return edges, np.unique(triplets, axis=0)


def compute_centroid(mol):
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    coords = [conf.GetAtomPosition(i) for i in range(num_atoms)]

    # 计算平均坐标
    x = sum(pos.x for pos in coords) / num_atoms
    y = sum(pos.y for pos in coords) / num_atoms
    z = sum(pos.z for pos in coords) / num_atoms

    return (x, y, z)



# ----------------------------
# 6. 数据集与批处理构建
# ----------------------------
class CrossDockDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        self.device = device
        self.stats_file = os.path.join(data_path, f'{prefix}_topo_stats.pkl')

        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)


        else:
            print(f'正在预处理数据集，前缀为 {prefix}')
            self.data = self.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    # 修改 __getitem__ 方法
    # def __getitem__(self, idx):
    #     """动态分层归一化 (完整版)"""
    #     item = self.data[idx]
    #     scaffold_mask = item['scaffold_mask'].bool().cpu().numpy()
    #     frag_mask = item['fragment_mask'].bool().cpu().numpy()
    #
    #     # 定义归一化处理器
    #     def normalize_feature(feature, region, mask):
    #         """通用归一化函数"""
    #         mean = self.topo_stats[region][feature]['mean']
    #         std = self.topo_stats[region][feature]['std'] + 1e-8
    #         return (item[f'topo_{feature}'][mask] - mean) / std
    #
    #     # 处理 betweenness 和 closeness 特征
    #     for feature in ['betweenness', 'closeness']:
    #         # 片段区域归一化
    #         frag_region_mask = scaffold_mask & frag_mask
    #         if frag_region_mask.any():
    #             item[f'topo_{feature}'][frag_region_mask] = normalize_feature(feature, 'frag', frag_region_mask)
    #
    #         # 剩余骨架区域归一化
    #         rem_region_mask = scaffold_mask & ~frag_mask
    #         if rem_region_mask.any():
    #             item[f'topo_{feature}'][rem_region_mask] = normalize_feature(feature, 'rem', rem_region_mask)
    #
    #     return item

    def preprocess(self, data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
        fragment_path = os.path.join(data_path, f'{prefix}_fragment.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')
        rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
        fenzi_path = os.path.join(data_path, f'{prefix}_mol.sdf')

        scaffold_mols = list(read_sdf(scaffold_path))
        fragment_mols = list(read_sdf(fragment_path))
        rgroup_mols = list(read_sdf(rgroups_path))
        full_mols = list(read_sdf(fenzi_path))
        table = pd.read_csv(table_path)

        with open(pockets_path, 'rb') as f:
            pocket_data_list = pickle.load(f)

        assert len(full_mols) == len(scaffold_mols) == len(fragment_mols) == len(rgroup_mols) == len(pocket_data_list) == len(table), "数据文件中的条目数不匹配！"

        MAX_GEO_FEATURES = 9  # 假设每个连接点提取3个距离特征×3个最近邻

        scaffold_atom_counts = [scaf_mol.GetNumAtoms() for scaf_mol in scaffold_mols]
        Nmax = max(scaffold_atom_counts)

        for idx in tqdm(range(len(full_mols)), desc="处理分子"):
            try:
                full_mol = full_mols[idx]
                scaf_mol = scaffold_mols[idx]
                frag_mol = fragment_mols[idx]
                rgroup_mol = rgroup_mols[idx]
                pocket_data = pocket_data_list[idx]
                row = table.iloc[idx]
                frag_uuid = row['uuid']

                # 解析骨架分子中片段和剩余部分
                frag_result = parse_scaffold_with_fragment(scaf_mol, frag_mol)
                if frag_result is None:
                    print(f"[{idx}] 无法解析骨架片段: uuid={frag_uuid}")
                    continue
                frag_pos, frag_hot, frag_charges, frag_degrees,frag_global_idx,frag_attach_rem_mask, rem_pos, rem_hot, \
                    rem_charges, rem_degrees,rem_global_idx,rem_attach_frag_mask, remaining_mask, attachment_bonds,scaf_edge_index_global,scaf_edge_index,scaf_bond_type_arr  = frag_result


                # 解析骨架与 R 基团的连接点
                attach_result = find_attachment_and_topology(full_mol, scaf_mol, rgroup_mol)
                if attach_result is None:
                    print(f"[{idx}] find_attachment_and_topology 失败: uuid={frag_uuid}")
                    continue
                rgroup_pos, rgroup_hot, rgroup_charges, rgroup_attach_scaf_mask, scaf_pos, scaf_attach_rgroup_mask, edge_index, bond_type_arr = attach_result

                frag_attach_rgroup_mask, rem_attach_rgroup_mask = map_scaf_attach_rgroup_to_frag_rem(
                    scaf_attach_rgroup_mask,
                    frag_global_idx,
                    rem_global_idx
                )

                # num_fake = 15
                # fake_scaf_pos = compute_centroid(scaf_mol)
                # # 使用与 R 基团相同的假原子填充机制，假原子用 '#' 表示
                # fake_pos = np.array( [fake_scaf_pos] * num_fake)  # 使用片段第一个原子的位置
                # fake_hot = np.array([get_one_hot('#', const.ATOM2IDX)] * num_fake)
                # fake_charges = np.array([const.CHARGES['#']] * num_fake)
                # fake_degrees = np.zeros(num_fake, dtype=np.float32)
                # # 填充到 rem 部分
                # rem_pos = np.concatenate([rem_pos, fake_pos])
                # rem_hot = np.concatenate([rem_hot, fake_hot])
                # rem_charges = np.concatenate([rem_charges, fake_charges])
                # rem_degrees = np.concatenate([rem_degrees, fake_degrees])
                # rem_attach_frag_mask = np.concatenate([rem_attach_frag_mask, np.zeros(num_fake)])
                # rem_attach_rgroup_mask = np.concatenate([rem_attach_rgroup_mask, np.zeros(num_fake)])

                fake_pos =frag_pos[0]
                rgroup_pos, rgroup_hot, rgroup_charges, rgroup_attach_scaf_mask= parse_rgroup_with_fake_atoms(
                    rgroup_pos, rgroup_hot, rgroup_charges, rgroup_attach_scaf_mask, fake_pos, max_atoms=10
                )

                # 处理口袋数据
                pocket_pos, pocket_hot, pocket_charges = [], [], []
                for i in range(len(pocket_data[f'{pocket_mode}_types'])):
                    atom_type = pocket_data[f'{pocket_mode}_types'][i]
                    pos = pocket_data[f'{pocket_mode}_coord'][i]
                    if atom_type == 'H':
                        continue
                    pocket_pos.append(pos)
                    pocket_hot.append(get_one_hot(atom_type, const.ATOM2IDX))
                    pocket_charges.append(const.CHARGES[atom_type])
                pocket_pos = np.array(pocket_pos)
                pocket_hot = np.array(pocket_hot)
                pocket_charges = np.array(pocket_charges)

                # 组合所有原子（不重复 scaf_pos）
                all_positions = np.concatenate([frag_pos, rem_pos, rgroup_pos, pocket_pos])
                all_one_hot = np.concatenate([frag_hot, rem_hot, rgroup_hot, pocket_hot])
                all_charges = np.concatenate([frag_charges, rem_charges, rgroup_charges, pocket_charges])
                all_degrees = np.concatenate([frag_degrees, rem_degrees, np.zeros(len(rgroup_pos)), np.zeros(len(pocket_pos))])


                all_attachment = np.concatenate([frag_attach_rem_mask, rem_attach_frag_mask, rgroup_attach_scaf_mask, np.zeros(len(pocket_pos))])
                all_attachment1 = np.concatenate(
                    [frag_attach_rgroup_mask, rem_attach_rgroup_mask, rgroup_attach_scaf_mask, np.zeros(len(pocket_pos))])

                # 生成掩码
                n_frag = len(frag_pos)
                n_rem = len(rem_pos)
                n_rgroup = len(rgroup_pos)
                n_pocket = len(pocket_pos)


                remaining_mask_full = np.concatenate([np.zeros(n_frag), np.ones(n_rem), np.zeros(n_rgroup + n_pocket)])
                fragment_mask = np.concatenate([np.ones(n_frag), np.zeros(n_rem + n_rgroup + n_pocket)])
                scaffold_mask = np.concatenate([np.ones(n_frag + n_rem), np.zeros(n_rgroup + n_pocket)])
                rgroup_mask = np.concatenate([np.zeros(n_frag + n_rem), np.ones(n_rgroup), np.zeros(n_pocket)])
                pocket_mask = np.concatenate([np.zeros(n_frag + n_rem + n_rgroup), np.ones(n_pocket)])
                scaffold_pockets_mask = np.concatenate([np.ones(n_frag + n_rem ),np.zeros(n_rgroup), np.ones(n_pocket)])
                #
                # # 计算键信息和三元组
                # edges, triplets = preprocess_bond_info(full_mol)
                # edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)  # (2, num_edges)
                # triplets = torch.tensor(triplets, dtype=torch.long, device=device)  # (num_angles, 3)
                #
                # # 计算键类型（可选）
                # bond_types = []
                # for bond in full_mol.GetBonds():
                #     bond_type = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(
                #         bond.GetBondType(), 1)
                #     bond_types.append(bond_type)
                # bond_types = torch.tensor(bond_types, dtype=torch.long, device=device) if bond_types else None

                assert len(all_positions) == len(all_one_hot) == len(all_charges) == len(all_attachment) == len(
                    all_degrees) == len(fragment_mask) == len(remaining_mask_full) == len(scaffold_mask) == len(
                    rgroup_mask) == len(pocket_mask) == len(all_attachment1), \
                    f"形状不匹配: all_positions={len(all_positions)}, masks={len(scaffold_mask)}"

                data.append({
                    'uuid': frag_uuid,
                    'positions': torch.tensor(all_positions, dtype=const.TORCH_FLOAT, device=device),
                    'one_hot': torch.tensor(all_one_hot, dtype=const.TORCH_FLOAT, device=device),
                    # 'charges': torch.tensor(all_charges, dtype=const.TORCH_FLOAT, device=device),
                    'attachment': torch.tensor(all_attachment, dtype=const.TORCH_FLOAT, device=device),
                    # 'degrees': torch.tensor(all_degrees, dtype=const.TORCH_FLOAT, device=device),
                    'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                    'rem_mask': torch.tensor(remaining_mask_full, dtype=const.TORCH_FLOAT, device=device),
                    'scaffold_mask': torch.tensor(scaffold_mask, dtype=const.TORCH_FLOAT, device=device),
                    'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
                    'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                    # 'scaffold_pockets_mask': torch.tensor(scaffold_pockets_mask, dtype=const.TORCH_FLOAT, device=device),
                    # 'attachment1': torch.tensor(all_attachment1, dtype=torch.float32, device=device),
                    'num_atoms': len(all_positions),
                    # 'scaf_edge_index':scaf_edge_index,
                    # 'scaf_bond_type_arr':scaf_bond_type_arr,
                    # 'edge_index': edge_index,
                    # 'triplets': triplets,
                    # 'bond_types': bond_types,
                })
            except Exception as e:
                print(f"错误索引: {idx}, uuid={frag_uuid}")
                try:
                    orig_smiles = Chem.MolToSmiles(full_mol)
                    scaf_smiles = Chem.MolToSmiles(scaf_mol)
                    print(f"完整分子: {orig_smiles}")
                    print(f"骨架分子: {scaf_smiles}")
                except:
                    pass
                print(f"异常信息: {e}")
                continue

        return data
def compute_adaptive_radius(points):
    """根据核心区域空间分布自动计算半径"""
    if len(points) == 0:
        return 0, 0
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_distance = np.max(distances)
    coarse_radius = max_distance + 3.0  # 粗粒度半径 = 核心区域半径 + 3Å 缓冲
    fine_radius = max_distance * 1.5    # 细粒度半径 = 核心区域半径 × 1.5
    return coarse_radius, fine_radius

def topology_aware_mask(positions, topo_features, core_points, base_radius):
    """结合拓扑重要性的掩码生成"""
    centroid = np.mean(core_points, axis=0) if len(core_points) > 0 else np.zeros(3)
    geom_dists = np.linalg.norm(positions - centroid, axis=1)
    geom_mask = (geom_dists < base_radius).astype(float)
    betweenness = topo_features['betweenness']
    closeness = topo_features['closeness']
    combined_weights = (
        0.5 * (1 - geom_dists / base_radius) +
        0.3 * betweenness +
        0.2 * closeness
    )
    threshold = np.percentile(combined_weights, 70)  # 动态阈值，取前30%权重区域
    topo_mask = (combined_weights > threshold).astype(float)
    return np.clip(geom_mask + topo_mask, 0, 1)


def generate_enhanced_masks(positions, core_points, topo_features):
    """生成改进的多尺度掩码"""
    coarse_radius, fine_radius = compute_adaptive_radius(core_points)

    # 基础几何掩码
    geom_coarse = distance_based_mask(positions, core_points, coarse_radius)
    geom_fine = distance_based_mask(positions, core_points, fine_radius)

    # 拓扑增强掩码
    topo_coarse = topology_aware_mask(positions, topo_features, core_points, coarse_radius)
    topo_fine = topology_aware_mask(positions, topo_features, core_points, fine_radius)

    # 融合掩码，确保层次性
    final_coarse = np.where(topo_coarse > 0.5, 1, geom_coarse)
    final_fine = np.where(topo_fine > 0.8, 1, geom_fine)

    return final_coarse, final_fine


def distance_based_mask(positions, core_points, radius):
    """基于距离生成基础掩码"""
    if len(core_points) == 0:
        return np.zeros(len(positions))
    core_center = np.mean(core_points, axis=0)
    dists = np.linalg.norm(positions - core_center, axis=1)
    return (dists < radius).astype(float)


def calculate_editable_mask(positions, attachment_points, radius=3.0):
    """生成可编辑区域掩码"""
    mask = np.zeros(len(positions))
    for pt in attachment_points:
        dists = np.linalg.norm(positions - positions[pt], axis=1)
        mask += (dists < radius).astype(float)
    return np.clip(mask, 0, 1)


def generate_multi_scale_mask(positions, core_points, radius_coarse, radius_fine):
    # 该函数根据原子与核心区域中心的距离，以及指定的两种半径（粗尺度和精细尺度），生成两个掩码
    """生成多尺度掩码"""
    coarse_mask = np.zeros(len(positions))
    fine_mask = np.zeros(len(positions))

    if len(core_points) == 0:
        return coarse_mask, fine_mask

    # 计算到核心区域的距离
    core_center = np.mean(core_points, axis=0)
    dists = np.linalg.norm(positions - core_center, axis=1)

    coarse_mask[dists < radius_coarse] = 1
    fine_mask[dists < radius_fine] = 1
    return coarse_mask, fine_mask


def collate(batch):
    """
    批处理函数：将每个键的数据填充到相同长度，并生成多套原子掩码和边掩码。

    参数:
        batch: 一个包含多个数据样本的列表，每个样本是一个字典。

    返回:
        out: 处理后的批次数据字典，包含填充后的张量和生成的掩码。
    """
    with torch.no_grad():
        out = {}


        # 收集每个键的数据
        for data in batch:
            for key, value in data.items():
                out.setdefault(key, []).append(value)

        # 对需要填充的属性进行统一处理
        for key in out.keys():
            if key in const.DATA_LIST_ATTRS:
                continue
            if key in const.DATA_ATTRS_TO_PAD:
                data = out[key]
                new_data = []
                for item in data:
                    if isinstance(item, np.ndarray):
                        new_data.append(torch.from_numpy(item))
                    else:
                        new_data.append(item)

                padded = torch.nn.utils.rnn.pad_sequence(new_data, batch_first=True, padding_value=0)

                out[key] = padded
            else:
                raise Exception(f'未知的批处理键: {key}')

        # atom_mask_1 = (out['scaffold_pockets_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
        atom_mask = (out['fragment_mask'].bool() |out['rem_mask'].bool()| out['rgroup_mask'].bool()| out['pocket_mask'].bool()).to(const.TORCH_INT)
        out['atom_mask'] = atom_mask[:, :, None]
        # out['atom_mask_1'] = atom_mask_1[:, :, None]

        # 生成不同阶段的原子掩码
        batch_size, n_nodes = atom_mask.size()


        if 'pocket_mask' in batch[0].keys():
            batch_mask = torch.cat([
                torch.ones(n_nodes, dtype=const.TORCH_INT) * i
                for i in range(batch_size)
            ]).to(atom_mask.device)
            out['edge_mask'] = batch_mask
        else:
            edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]  # 形状：(batch_size, max_num_atoms, max_num_atoms)
            diag_mask = ~torch.eye(n_nodes, dtype=torch.int, device=atom_mask.device).unsqueeze(0)
            edge_mask *= diag_mask  # 去除对角线（自连接）
            out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
            if key in out.keys():
                out[key] = out[key][:, :, None]
        return out

        # batch_size = len(batch)
        # device = out['positions'].device
        # # 为每个部分生成 pos、atom_feature 和 batch_mask
        # parts = ['pocket', 'scaffold', 'fragment', 'rgroup','rem']
        # for part in parts:
        #     part_mask = out[f'{part}_mask'].bool() & out['atom_mask'].bool().squeeze(-1)
        #     part_pos_list = []
        #     part_atom_feature_list = []
        #     part_batch_indices = []
        #
        #     for i in range(batch_size):
        #         # 获取第 i 个样本的掩码
        #         sample_part_mask = part_mask[i]
        #         # 提取第 i 个样本的坐标和特征
        #         sample_part_pos = out['positions'][i][sample_part_mask]
        #         sample_part_atom_feature = out['one_hot'][i][sample_part_mask]
        #         # 生成批次索引
        #         num_part_atoms = sample_part_pos.size(0)
        #         sample_batch = torch.full((num_part_atoms,), i, dtype=torch.long, device=device)
        #         # 添加到列表
        #         part_pos_list.append(sample_part_pos)
        #         part_atom_feature_list.append(sample_part_atom_feature)
        #         part_batch_indices.append(sample_batch)
        #
        #     # 拼接所有样本的结果
        #     out[f'{part}_pos'] = torch.cat(part_pos_list, dim=0)
        #     out[f'{part}_atom_feature'] = torch.cat(part_atom_feature_list, dim=0)
        #     out[f'{part}_batch_mask'] = torch.cat(part_batch_indices, dim=0)

from model.categorical import Categorical
class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def __repr__(self):
        return f"DistributionNodes(n_nodes={self.n_nodes.tolist()}, prob={self.prob.tolist()})"


    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs
def create_template_scaffold(tensor, frag_size, rem_size, rgroup_size, pocket_size, fill=0):
    """
    根据输入 tensor 及各部分尺寸构造模板。

    参数：
      tensor: 输入张量，形状为 [N, feature_dim]（例如 [N,3]），
              N 应等于 frag_size + rem_size + rgroup_size + pocket_size，
              其中各部分按顺序依次排列。
      frag_size: 片段部分的大小，需要保留原始值。
      rem_size: 剩余部分的大小，用填充值填充。
      rgroup_size: R 基团部分的大小，用填充值填充。
      pocket_size: 蛋白质口袋部分的大小，需要保留原始值。
      fill: 用于填充 rem 和 rgroup 部分的数值（默认为 0）。

    返回：
      一个新张量，其形状为 [frag_size + rem_size + rgroup_size + pocket_size, feature_dim]，
      其中前 frag_size 行和最后 pocket_size 行与输入相同，
      而中间 rem_size 和 rgroup_size 部分全部填充为 fill。
    """

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)  # 变为 [N, 1]
    # 保留前 frag_size 个

    frag = tensor[:frag_size]

    # 为剩余部分创建填充值
    rem_template = torch.full((rem_size, tensor.shape[1]), fill, dtype=tensor.dtype, device=tensor.device)

    # 为 R 基团部分创建填充值
    rgroup_template = torch.full((rgroup_size, tensor.shape[1]), 0, dtype=tensor.dtype, device=tensor.device)

    # 假设口袋部分位于最后 pocket_size 个位置
    pocket = tensor[-pocket_size:]

    # 拼接成最终模板
    return torch.cat([frag, rem_template, rgroup_template, pocket], dim=0)


def create_template(tensor, frag_size, rem_size, rgroup_size, pocket_size, fill=0):
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)  # 变为 [N, 1]
    # 保留前 frag_size 个

    scaf_size = frag_size + rem_size
    scaf = tensor[:scaf_size]

    # 为 R 基团部分创建填充值
    rgroup_template = torch.full((rgroup_size, tensor.shape[1]), fill, dtype=tensor.dtype, device=tensor.device)

    # 假设口袋部分位于最后 pocket_size 个位置
    pocket = tensor[-pocket_size:]

    # 拼接成最终模板
    return torch.cat([scaf, rgroup_template, pocket], dim=0)

def create_template_scaffold_r(tensor, frag_size, rem_size, rgroup_size, pocket_size, fill=0):
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)  # 变为 [N, 1]
    # 保留前 frag_size 个

    frag = tensor[:frag_size]

    # 为剩余部分创建填充值
    rem_template = torch.full((rem_size, tensor.shape[1]), 0, dtype=tensor.dtype, device=tensor.device)

    # 为 R 基团部分创建填充值
    rgroup_template = torch.full((rgroup_size, tensor.shape[1]), 1, dtype=tensor.dtype, device=tensor.device)

    # 假设口袋部分位于最后 pocket_size 个位置
    pocket = tensor[-pocket_size:]

    # 拼接成最终模板
    return torch.cat([frag, rem_template, rgroup_template, pocket], dim=0)
def create_template_rgroup(tensor, scaffold_size, rgroup_size, pocket_size, fill=0):

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)  # 变为 [N, 1]
    # 保留前 frag_size 个
    values_to_keep = tensor[:scaffold_size]

    rgroup_template = torch.full((rgroup_size, tensor.shape[1]), fill, dtype=tensor.dtype, device=tensor.device)

    # 假设口袋部分位于最后 pocket_size 个位置
    pocket = tensor[-pocket_size:]

    # 拼接成最终模板
    return torch.cat([values_to_keep ,rgroup_template, pocket], dim=0)

def sample_local_atom_counts(n_list, delta=15, lower=5, upper=60):
    return [int(truncnorm((lower - n) / delta, (upper - n) / delta, loc=n, scale=delta).rvs()) for n in n_list]


def create_templates_for_rgroup_generation_single(data, sizes, id):

    if id == 0:
        print(sizes)


        nodes_dist = DistributionNodes(const.n_nodes_pro)

        nodesxsample_list = []

        for i, rem_size in enumerate(sizes):
            nodesxsample = nodes_dist.sample(1).item()
            nodesxsample_list.append(nodesxsample)




        decoupled_data = []
        for i, rem_size in enumerate(sizes):
            print(f"Sample {i}: rem_size = {rem_size}")
            rem_size = nodesxsample_list[i]
            data_dict = {}
            try:
                fragment_mask = data['fragment_mask'][i].squeeze()

                fragment_size = fragment_mask.sum().int()
                # print(fragment_size)

                pocket_mask= data['pocket_mask'][i].squeeze()
                pocket_size = pocket_mask.sum().int()
                # print(pocket_size)

                rgroup_mask = data['rgroup_mask'][i].squeeze()
                rgroup_size = rgroup_mask.sum().int()
                # print(rgroup_size)

            except Exception as e:
                print(f"Error processing 'fragment_mask' for sample {i}: {e}")
                raise
            for k, v in data.items():
                try:
                    if k == 'num_atoms':
                        data_dict[k] = fragment_size + rem_size  # 更新原子总数：骨架原子数 + R基团原子数
                        continue
                    if k in const.DATA_LIST_ATTRS:  # 不需要填充的属性直接复制
                        data_dict[k] = v[i]
                        continue
                    if k in const.DATA_ATTRS_TO_PAD:
                        fill_value = 1 if k in ['rem_mask','scaffold_mask'] else 0

                        template = create_template_scaffold(v[i], fragment_size, rem_size,rgroup_size,pocket_size, fill=fill_value)

                        if k == 'rgroup_mask':
                            template = create_template_scaffold_r(v[i], fragment_size, rem_size, rgroup_size=rgroup_size, pocket_size=pocket_size,
                                                                fill=1)
                        if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                            template = template.squeeze(-1)
                        data_dict[k] = template
                        # if k == 'rgroup_mask':
                        #     print(data_dict[k].shape)
                        #     print(data_dict[k])
                except Exception as e:
                    print(f"Error processing key '{k}' for sample {i}: {e}")
                    raise
            decoupled_data.append(data_dict)
        return collate(decoupled_data)
    else:
        rgroup_sizes = sizes
        decoupled_data = []
        for i, rgroup_size in enumerate(rgroup_sizes):
            data_dict = {}
            try:

                scaffold_mask = data['scaffold_mask'][i].squeeze()
                scaffold_size = scaffold_mask.sum().int()

                pocket_mask = data['pocket_mask'][i].squeeze()
                pocket_size = pocket_mask.sum().int()

                # print("r基团-------")
                # print(scaffold_size)
            except Exception as e:
                print(f"Error processing 'scaffold_mask' for sample {i}: {e}")
                raise
            for k, v in data.items():
                try:
                    if k == 'num_atoms':
                        data_dict[k] = scaffold_size + rgroup_size  # 更新原子总数：骨架原子数 + R基团原子数
                        continue
                    if k in const.DATA_LIST_ATTRS:  # 不需要填充的属性直接复制
                        data_dict[k] = v[i]
                        continue
                    if k in const.DATA_ATTRS_TO_PAD:
                        fill_value = 1 if k =='rgroup_mask' else 0  # 对于 rgroup_mask 填充 1，其它填充 0
                        # template = create_template(v[i], fragment_size,rem_size, rgroup_size,pocket_size,fill=fill_value)
                        template = create_template_rgroup(v[i], scaffold_size, rgroup_size, pocket_size,fill=fill_value)

                        if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                            template = template.squeeze(-1)
                        data_dict[k] = template
                    # if k == 'rgroup_mask':
                    #     print(data_dict[k].shape)
                    #     print(data_dict[k])

                except Exception as e:
                    print(f"Error processing key '{k}' for sample {i} (rgroup branch): {e}")
                    raise
            # print("fragment_mask shape:", data_dict['fragment_mask'].shape)
            # print("rem_mask shape:", data_dict['rem_mask'].shape)
            # print("rgroup_mask shape:", data_dict['rgroup_mask'].shape)
            # print("pocket_mask shape:", data_dict['pocket_mask'].shape)
            # atom_mask = (data_dict['fragment_mask'].bool() |data_dict['rem_mask'].bool() | data_dict['rgroup_mask'].bool() | data_dict[
            #     'pocket_mask'].bool()).to(const.TORCH_INT)
            # print(atom_mask.shape)
            decoupled_data.append(data_dict)
        return collate(decoupled_data)



def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False,num_workers=0, pin_memory=False,sampler=None):
    """创建DataLoader"""
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle,num_workers=num_workers, sampler=sampler,pin_memory=pin_memory)