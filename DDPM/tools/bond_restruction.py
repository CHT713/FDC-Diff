import subprocess
import time

import torch
import numpy as np
from rdkit import Chem, Geometry
import os
import sascorer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import AllChem

MARGINS_EDM = [10, 5, 2]

BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def generate_mappings(items_list):
    item2idx = {item: idx for idx, item in enumerate(items_list)}
    idx2item = {idx: item for idx, item in enumerate(items_list)}

    return item2idx, idx2item


ALLOWED_ATOM_TYPES = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P']
ATOM2IDX, IDX2ATOM = generate_mappings(ALLOWED_ATOM_TYPES)


BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}
# ==========================================
# 2. 你的原始核心逻辑 (稍作封装以适配 Class)
# ==========================================

def get_bond_order(atom1, atom2, distance, check_exists=True, margins=MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in BONDS_1:
            return 0
        if atom2 not in BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        # if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            # thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            # if distance < thr_bond2:
            #     if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
            #         thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
            #         if distance < thr_bond3:
            #             return 3  # Triple
            #     return 2  # Double
        return 1  # Single
    return 0  # No bond



def build_xae_molecule(positions, atom_types, margins=MARGINS_EDM):
    """
    Args:
        positions: N x 3 (Tensor)
        atom_types: N (Tensor, int)
    """
    n = positions.shape[0]
    X = atom_types

    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    idx2atom = IDX2ATOM

    # 计算距离矩阵
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dists[i, j], margins=margins)

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order

    return X, A, E


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def build_molecule(positions, atom_types, margins=MARGINS_EDM):
    idx2atom = IDX2ATOM

    # 1. 推断连接关系 (X:原子类型, A:邻接矩阵, E:边类型)
    X, A, E = build_xae_molecule(positions, atom_types, margins=margins)

    mol = Chem.RWMol()

    # 2. 添加原子
    for atom in X:
        a = Chem.Atom(idx2atom[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), BOND_DICT[E[bond[0], bond[1]].item()])

    mol.AddConformer(create_conformer(positions.detach().cpu().numpy().astype(np.float64)))

    return mol


# ==========================================
# 3. XYZ 读取与主程序
# ==========================================

def read_xyz_and_build(xyz_file_path):
    """ 读取 XYZ 文件并构建分子 """

    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()

    # 简单解析逻辑
    atoms = []
    coords = []

    start_line = 0
    # 判断数据从哪一行开始
    if len(lines[0].split()) == 1 and lines[0].strip().isdigit():
        # 第一行是数字，标准格式
        # 检查第二行是否是原子数据
        tokens_line2 = lines[1].split()
        if len(tokens_line2) >= 4 and tokens_line2[0].isalpha():
            # 第二行是数据（缺少注释行的情况）
            start_line = 1
        else:
            # 第二行是注释，第三行是数据
            start_line = 2

    # 解析原子信息
    for line in lines[start_line:]:
        tokens = line.strip().split()
        if len(tokens) < 4:
            continue

        symbol = tokens[0]
        # 将符号转换为原子序号 (Atom Types)
        if symbol in ATOM2IDX:
            type_idx = ATOM2IDX[symbol]
        else:
            print(f"Warning: Unknown atom symbol {symbol}, skipping.")
            continue

        x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])

        atoms.append(type_idx)
        coords.append([x, y, z])

    # 转换为 Tensor
    positions_tensor = torch.tensor(coords, dtype=torch.float32)

    # 修改 atom_types_tensor 为存储原子类型索引的 1D tensor
    atom_types_tensor = torch.tensor(atoms, dtype=torch.long)

    print(f"Read {len(atoms)} atoms from XYZ.")

    # 调用核心构建函数
    mol = build_molecule(positions_tensor, atom_types_tensor)
    return mol

def is_valid_molecule(sdf_path):
    """
    Check if a molecule in an SDF file is valid using RDKit.
    - Ensures the molecule is a single connected fragment.
    - Verifies that atomic valences are chemically reasonable.

    Args:
        sdf_path (str): Path to the SDF file.
    Returns:
        bool: True if the molecule is valid, False otherwise.
    """
    try:
        # Load the SDF file
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next(suppl)
        if mol is None:
            print(f"Failed to load molecule from {sdf_path}.")
            return False

        # Check connectivity: ensure only one fragment
        fragments = Chem.GetMolFrags(mol)
        if len(fragments) > 1:
            print(f"Molecule in {sdf_path} has multiple fragments.")
            return False

        # Check valence/bond validity
        AllChem.ComputeGasteigerCharges(mol)  # Implicitly validates valences
        for atom in mol.GetAtoms():
            if atom.GetNumImplicitHs() < 0:
                print(f"Atom {atom.GetIdx()} in {sdf_path} has invalid valence.")
                return False

        return True

    except Exception as e:
        print(f"Error processing {sdf_path}: {e}")
        return False





def get_sa_score_from_sdf(sdf_path):
    mol = Chem.SDMolSupplier(sdf_path, sanitize=True)[0]
    if mol is None:
        return None
    sa_score = sascorer.calculateScore(mol)
    sa_norm = round((10 - sa_score) / 9, 2)
    return sa_norm


def calculate_qed_from_single_sdf(sdf_path):
    """
    读取一个只包含一个分子的 .sdf 文件，计算其 QED 分数。

    参数:
        sdf_path (str): 单分子 SDF 文件路径。

    返回:
        float 或 None: 返回 QED 分数，若失败则返回 None。
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0] if suppl and suppl[0] is not None else None

    if mol is not None:
        try:
            return QED.qed(mol)
        except:
            return None
    return None


def yuel_bond(out_xyz,out_sdf):
    yuel_bond_path = 'yuel_bond/yuel_bond.py'  # yuel_bond.py 的路径
    model_path = 'yuel_bond/models/geom_3d.ckpt'  # 模型路径
    model_path_cdg = 'yuel_bond/models/geom_cdg.ckpt'  # 模型路径
    result = subprocess.run(
        f'python {yuel_bond_path} {out_xyz} {out_sdf} --model {model_path}',  # 替换为你的python环境路径
        shell=True, capture_output=True, text=True
    )

    # 检查 yuel_bond.py 转换后的 sdf 是否有效
    if os.path.exists(out_sdf) and is_valid_molecule(out_sdf) > 0:

       
        all_valid = True  #

    else:
        
        if os.path.exists(out_sdf):
            os.remove(out_sdf)
          
        result2 = subprocess.run(
            f'python {yuel_bond_path} {out_xyz} {out_sdf} --model {model_path_cdg}',   # 替换为你的python环境路径
            shell=True, capture_output=True, text=True
        )

        if os.path.exists(out_sdf) and is_valid_molecule(out_sdf):
         
            all_valid=True
        else:
    
            os.remove(out_xyz)
            os.remove(out_sdf)
            all_valid = False

    return all_valid

# def openbabel_bond(out_xyz,out_sdf):
#     obabel_path = 'obabel'  # 换成你的 obabel 路径


#     # 第二步：将优化后的 xyz 转换为 sdf
#     result = subprocess.run(
#         f'{obabel_path} {out_xyz} -O {out_sdf} --minimize --ff MMFF94',
#         shell=True, capture_output=True, text=True
#     )
#     if os.path.exists(out_sdf) and is_valid_molecule(out_sdf):
#         all_valid=True
#     else:
#         os.remove(out_xyz)
#         os.remove(out_sdf)
#         all_valid = False

#     return  all_valid


def bond_CN(out_xyz,output_dir,pred_names_r,j,success_count,retries):
    out_sdf_yb = f'{output_dir}/{pred_names_r[j]}_y.sdf'
    out_sdf_rd = f'{output_dir}/{pred_names_r[j]}_r.sdf'

    all_valid_yb=yuel_bond(out_xyz,out_sdf_yb)

    if all_valid_yb:
        sa_yb = get_sa_score_from_sdf(out_sdf_yb)
        qed_yb = calculate_qed_from_single_sdf(out_sdf_yb)
        yb_score=sa_yb+qed_yb
    elif retries<5:
        return   out_sdf_yb,success_count, all_valid_yb
    else:
        yb_score = 0

    
    mol = read_xyz_and_build(out_xyz)
    Chem.SanitizeMol(mol)
    # 保存为 SDF 供查看
    w = Chem.SDWriter(out_sdf_rd)
    w.write(mol)
    w.close()
    if os.path.exists(out_sdf_rd) and is_valid_molecule(out_sdf_rd):
        all_valid_rd = True
        sa_rd = get_sa_score_from_sdf(out_sdf_rd)
        qed_rd = calculate_qed_from_single_sdf(out_sdf_rd)
        rd_score = sa_rd + qed_rd
    else:
        all_valid_rd=False
        rd_score = 0


    print(yb_score,rd_score)
    best_score = max(yb_score, rd_score)
    if best_score == yb_score:
        best_sdf = out_sdf_yb
    else:
        best_sdf = out_sdf_rd

   
    if best_sdf != out_sdf_yb:
        os.remove(out_sdf_yb)
    if best_sdf != out_sdf_rd:
        os.remove(out_sdf_rd)

    out_sdf = best_sdf
    all_valid = all_valid_yb or all_valid_rd
    if all_valid:
        success_count += 1

    return out_sdf,success_count, all_valid
# ==========================================
# 4. 运行示例
# ==========================================

if __name__ == "__main__":
    # 为了演示，我先把你提供的 XYZ 数据写入一个临时文件
    xyz_content = """43
C 41.750999451 35.861000061 84.039001465
C 40.668998718 36.643001556 83.623001099
C 40.157001495 37.665000916 84.427001953
C 40.696998596 37.929000854 85.700996399
C 41.784999847 37.162998199 86.132003784
C 42.332000732 36.147998810 85.286003113
N 43.725990295 41.189338684 84.780364990
C 44.812694550 36.038448334 84.990539551
N 44.808673859 37.300559998 84.927726746
O 46.323013306 35.245357513 81.806106567
C 45.773891449 35.845687866 82.932334900
C 44.818950653 34.404907227 91.892341614
C 44.763317108 38.213047028 80.429664612
C 45.114837646 36.318462372 90.705169678
C 42.957313538 35.562469482 90.115058899
N 45.441402435 35.393264771 84.149749756
C 43.371578217 41.139514923 83.450950623
C 43.499130249 34.894023895 87.744033813
C 43.986927032 35.482170105 91.259201050
N 43.769466400 35.439350128 88.944412231
O 46.575202942 36.038780212 79.638725281
C 45.146335602 39.708602905 86.180000305
N 43.791339874 40.007472992 82.848785400
N 44.326240540 35.801998138 86.797042847
C 43.583679199 35.303180695 85.670761108
C 46.144088745 36.122486115 80.803062439
C 44.987815857 36.210750580 89.274108887
C 44.399192810 40.062042236 84.995864868
C 44.400726318 39.271652222 83.819923401
C 45.293441772 37.146728516 82.667701721
C 44.889469147 37.911144257 83.763542175
C 45.793479919 39.497985840 87.221305847
O 42.535423279 34.009750366 87.629615784
C 45.351970673 37.191329956 81.221847534
C 39.689903259 37.936458588 89.597213745
C 39.033836365 38.395935059 87.405654907
C 39.004394531 39.300071716 88.585296631
N 38.584732056 40.597446442 88.285903931
C 37.419986725 41.343788147 88.773979187
C 40.259609222 40.323974609 86.463249207
C 39.376106262 41.191635132 87.275299072
O 40.019931793 39.032646179 89.449378967
C 40.070472717 38.895225525 86.573387146
"""
    filename = "input.xyz"
    with open(filename, "w") as f:
        f.write(xyz_content)

    # 执行构建
    try:
        mol = read_xyz_and_build(filename)

        # 输出结果验证
        print("\nMolecule Built Successfully!")
        print(f"Num Atoms: {mol.GetNumAtoms()}")
        print(f"Num Bonds: {mol.GetNumBonds()}")

        # 生成 SMILES 看看结构是否合理
        # 注意：由于原逻辑中没有进行 Sanitize 或 Valency 检查，
        # 且只生成单键，SMILES 可能会包含电荷或不完整的芳香环，这是预期行为。
        try:
            Chem.SanitizeMol(mol)
            print("SMILES:", Chem.MolToSmiles(mol))

            # 保存为 SDF 供查看
            w = Chem.SDWriter('output.sdf')
            w.write(mol)
            w.close()
            print("Saved to output.sdf")
        except Exception as e:
            print("Sanitization failed (expected for distance-based bond inference w/o bond orders):", e)
            print("Raw SMILES:", Chem.MolToSmiles(mol, sanitize=False))

    except Exception as e:
        print("Error building molecule:", e)
        import traceback

        traceback.print_exc()

    # 清理临时文件
    if os.path.exists(filename):
        os.remove(filename)
