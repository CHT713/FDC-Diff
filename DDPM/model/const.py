import torch

from rdkit import Chem

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

# Atom idx for one-hot encoding '#'-fake atom
ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, '#': 9}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P', 9: '#'}

# Atomic numbers
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15, '#': 0}
CHARGES_LIST = [6, 8, 7, 9, 16, 17, 35, 53, 15, 0]

# One-hot atom types
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)

DATA_LIST_ATTRS = {'uuid', 'name', 'scaffold_smi', 'rgroup_smi', 'num_atoms',
                   'cat', 'rgroup_size', 'anchors_str', 'edge_index', 'scaf_edge_index',
                    'scaf_bond_type_arr', 'triplets',
                     'bond_types'
                    'edge_index',
                    'bond_types'}

# 定义需要填充的属性
DATA_ATTRS_TO_PAD = {'positions', 'one_hot', 'charges', 'anchors',
                     'fragment_mask', 'rem_mask', 'scaffold_mask',
                     'rgroup_mask', 'pocket_mask', 'scaffold_pockets_mask',
                     'topo_betweenness', 'topo_closeness', 'geo_features',
                     'coarse_mask', 'fine_mask', 'editable_mask', 'degrees', 'attachment','attachment1'
                    }

# 定义需要增加最后一维的属性
DATA_ATTRS_TO_ADD_LAST_DIM = {'charges', 'anchors', 'scaffold_mask', 'rgroup_mask', 'pocket_mask', 'rem_mask',
                              'fragment_mask', 'scaffold_pockets_mask','attachment','attachemnt1'}

# 定义浮点型掩码属性（需要转换为布尔型）
FLOAT_MASK_ATTRS = {'fragment_mask', 'rem_mask', 'scaffold_mask',
                    'rgroup_mask', 'pocket_mask', 'scaffold_pockets_mask',
                    'coarse_mask', 'fine_mask', 'editable_mask'}

MARGINS_EDM = [10, 5, 2]
n_nodes_pro={13: 3435, 11: 2834, 30: 573, 18: 2534, 14: 3880, 8: 3024, 3: 756, 2: 841, 10: 3101, 7: 3438, 24: 1525, 26: 1421, 5: 1228, 12: 3693, 25: 938, 4: 594, 17: 2729, 20: 2638, 27: 1149, 19: 2544, 15: 3556, 28: 1203, 9: 2330, 1: 269, 33: 302, 32: 384, 22: 1414, 29: 571, 6: 2054, 21: 1960, 16: 3195, 35: 172, 31: 455, 23: 1536, 36: 156, 34: 93, 40: 85, 41: 22, 38: 115, 37: 83, 39: 50, 42: 98, 47: 10, 56: 5, 55: 1, 59: 2, 44: 8, 49: 4, 46: 3, 54: 5, 53: 3, 43: 3, 48: 5, 51: 3, 45: 5, 73: 1, 60: 7, 50: 1, 52: 4, 57: 1, 58: 2}
# ================================================================
# 例子：常见单键的平衡键长(Å)和“劲度系数”k_val(可视为约束强度)
# 这些数值供演示，可根据需要自行改动、扩展（包含双键、三键等）
# ================================================================
# 长度单位: Å
# k_val 是一个演示用数值, 依据键能强弱做大致调节, 并不严格等价(kJ/mol)/Å^2.
CHEMICAL_BOND_PARAMS = {
    # 单键
    ('H','H'): (0.74, 400.0),
    ('C','H'): (1.09, 300.0),
    ('C','C'): (1.54, 250.0),
    ('N','H'): (1.01, 280.0),
    ('N','N'): (1.45, 120.0),
    ('O','H'): (0.96, 350.0),
    ('O','O'): (1.48, 100.0),
    ('F','F'): (1.42, 120.0),
    ('Cl','Cl'): (1.99, 180.0),
    ('Br','Br'): (2.28, 150.0),
    ('I','I'): (2.67, 120.0),
    ('C','O'): (1.43, 250.0),
    ('C','N'): (1.47, 220.0),

    # 示例: 双键
    ('C','C_double'): (1.34, 400.0),  # C=C
    ('C','O_double'): (1.23, 500.0),  # C=O
    ('O','O_double'): (1.21, 350.0),  # O=O
    ('N','N_double'): (1.25, 400.0),  # 如果你有N=N双键, 可自行修正

    # 示例: 三键
    ('C','C_triple'): (1.20, 500.0),  # C≡C
    ('N','N_triple'): (1.10, 650.0),  # N≡N
    # 若还有更多可自行加
}
