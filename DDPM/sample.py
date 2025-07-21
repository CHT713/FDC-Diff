
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 确保只使用GPU 1
import argparse
import os
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm
import subprocess
import time
from model import function
from edm import EDM
from datasets import CrossDockDataset, collate, get_dataloader
from DDPMlast import DDPM  # 假设 DDPM 类的定义在 DDPM.py 中
from model import visualizer as vis



p = argparse.ArgumentParser()
p.add_argument('--config', type=str, default=None)
p.add_argument('--data', action='store', type=str, default="datasets")
p.add_argument('--train_data_prefix', action='store', type=str, default='train')
p.add_argument('--val_data_prefix', action='store', type=str, default='val')
p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
p.add_argument('--logs', action='store', type=str, default='logs')
p.add_argument('--device', action='store', type=str, default='cpu')
p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
p.add_argument('--log_iterations', action='store', type=str, default=20)
p.add_argument('--exp_name', type=str, default='YourName')
p.add_argument('--model', type=str, default='egnn_dynamics',
               help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')
p.add_argument('--diffusion_steps', type=int, default=1000)
p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
p.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')
p.add_argument('--n_epochs', type=int, default=200)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--brute_force', type=eval, default=False, help='True | False')
p.add_argument('--actnorm', type=eval, default=True, help='True | False')
p.add_argument('--break_train_epoch', type=eval, default=False, help='True | False')
p.add_argument('--dp', type=eval, default=True, help='True | False')
p.add_argument('--condition_time', type=eval, default=True, help='True | False')
p.add_argument('--clip_grad', type=eval, default=True, help='True | False')
p.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
p.add_argument('--n_layers', type=int, default=6, help='number of layers')
p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
p.add_argument('--nf', type=int, default=128, help='number of layers')
p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
p.add_argument('--norm_constant', type=float, default=1, help='diff/(|diff| + norm_constant)')
p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
p.add_argument('--ode_regularization', type=float, default=1e-3)
p.add_argument('--dataset', type=str, default='crossdock', help='crossdock')
p.add_argument('--datadir', type=str, default='/crossdock/', help='crossdock directory')
p.add_argument('--filter_n_atoms', type=int, default=None, help='')
p.add_argument('--dequantization', type=str, default='argmax_variational',
               help='uniform | variational | argmax_variational | deterministic')
p.add_argument('--n_report_steps', type=int, default=1)
p.add_argument('--wandb_usr', type=str)
p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
p.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
p.add_argument('--save_model', type=eval, default=True, help='save model')
p.add_argument('--generate_epochs', type=int, default=1, help='save model')
p.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
p.add_argument('--test_epochs', type=int, default=1)
p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
p.add_argument('--resume', type=str, default=None, help='')
p.add_argument('--start_epoch', type=int, default=0, help='')
p.add_argument('--ema_decay', type=float, default=0.999,
               help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
p.add_argument('--augment_noise', type=float, default=0)
p.add_argument('--n_stability_samples', type=int, default=500, help='Number of samples to compute the stability')
p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
               help='normalize factors for [x, categorical, integer]')
p.add_argument('--remove_h', action='store_true')
p.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
p.add_argument('--visualize_every_batch', type=int, default=1e8,
               help="Can be used to visualize multiple times per epoch")
p.add_argument('--normalization_factor', type=float, default=1, help="Normalize the sum aggregation of EGNN")
p.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')
p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
p.add_argument('--center_of_mass', type=str, default='scaffold', help='Where to center the data: scaffold | anchors')
p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')
p.add_argument('--cuda', type=bool, default=True)
p.add_argument('--logdir', type=str, default='./logs')

# 直接定义 args

def generate_animation(self, chain, node_mask, id):
        # 根据 id 设置保存路径
        if id == 'scaffold':
            animation_dir = os.path.join(self.samples_dir, 'scaffold_animation')
        else:
            animation_dir = os.path.join(self.samples_dir, 'rgroup_animation')
        os.makedirs(animation_dir, exist_ok=True)

        # 检查 chain 的格式
        if isinstance(chain, list):
            # 如果 chain 是列表，每个元素是一个帧
            for frame_idx, frame in enumerate(chain):
                x = frame[:, :, :self.edm.n_dims]  # 提取位置
                h = frame[:, :, self.edm.n_dims:]  # 提取特征
                name = f'{id}_frame_{frame_idx}'
                vis.save_xyz_file_fa(animation_dir, h, x, node_mask, names=[name])
        else:
            # 如果 chain 是单个张量，视为只有一帧
            x = chain[:, :, :self.edm.n_dims]
            h = chain[:, :, self.edm.n_dims:]
            name = f'{id}_frame_0'
            vis.save_xyz_file_fa(animation_dir, h, x, node_mask, names=[name])
# 检查已生成样本
def check_if_generated(output_dir, uuids, n_samples):
    generated = True
    starting_points = []
    for uuid in uuids:
        uuid_dir = os.path.join(output_dir, uuid)
        numbers = []
        if os.path.exists(uuid_dir):
            for fname in os.listdir(uuid_dir):
                try:
                    num = int(fname.split('_')[0])
                    numbers.append(num)
                except ValueError:
                    continue
        if not numbers or max(numbers) < n_samples - 1:
            generated = False
            starting_points.append(max(numbers) + 1 if numbers else 0)
    starting = min(starting_points) if starting_points else 0
    return generated, starting


from rdkit import Chem
from rdkit.Chem import AllChem


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
from sascorer import calculateScore
def get_sa_score_from_sdf(sdf_path):
    mol = Chem.SDMolSupplier(sdf_path, sanitize=True)[0]
    if mol is None:
        return None
    sa_score = calculateScore(mol)
    sa_norm = round((10 - sa_score) / 9, 2)
    return sa_norm


from rdkit import Chem
from rdkit.Chem import QED

from rdkit import Chem
from rdkit.Chem import QED

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



def two_stage_sampling(model, data, output_dir, uuids, n_samples, starting_point):

    for i in tqdm(range(starting_point, n_samples), desc="Sampling molecules"):
        retries = 0
        while retries < 1000:
            try:
                # 生成骨架
                chain_scaffold, node_mask_scaffold, num_atoms_scaffold,sample_atom_mask,sample_pocket_mask,sample_rgroup_mask,data_sample = model.sample_chain(
                    data=data, sample_fn=None, keep_frames=1, id=0
                )

                x_scaffold = chain_scaffold[-1][:, :, :model.n_dims]
                h_scaffold = chain_scaffold[-1][:, :, model.n_dims:]
                #

                center_of_mass_list=[]
                fragment_mask = data['fragment_mask']
                com_mask =fragment_mask
                x =data['positions']
                x_masked = x * com_mask
                N = com_mask.sum(1, keepdims=True)
                mean = torch.sum(x_masked, dim=1, keepdim=True) / N
                for j in range(mean.shape[0]):  # 计算得到的 COM 存入 center_of_mass_list，方便后续添加回生成的分子坐标中，保证生成分子的整体位置正确。
                    center_of_mass_list.append(mean[j][0].cpu().numpy().tolist())

                # # 保存骨架
                # pred_names = [f'{uuid}/{i}_scaffold' for uuid in uuids]

                x_scaffold += mean  # Adding COM

                # node_mask = data['atom_mask'] - data['pocket_mask']-data['rgroup_mask']
                # vis.save_xyz_file_fa(output_dir, h_scaffold, x_scaffold, node_mask, pred_names)
                #
                # for j in range(len(pred_names)):
                #     out_xyz = f'{output_dir}/{pred_names[j]}_.xyz'
                #     out_sdf = f'{output_dir}/{pred_names[j]}_.sdf'
                #     subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)
                # node_mask_s = sample_atom_mask - sample_pocket_mask - sample_rgroup_mask
                # pred_names_scaffold = [f'{uuid}/{i}_scaffold' for uuid in uuids]
                # vis.save_xyz_file_fa(output_dir, h_scaffold, x_scaffold, node_mask_s, pred_names_scaffold)


                # 准备 R 基团生成的数据
                generated_scaffold = {
                    'positions': x_scaffold,
                    'one_hot': h_scaffold
                    # 如果需要其他字段（如掩码），在这里添加
                }


                chain_rgroup, node_mask_rgroup, num_atoms_rgroup = model.sample_chain(
                    data=data_sample, sample_fn=None, keep_frames=1, id=1, generated_scaffold=generated_scaffold
                )


                x_rgroup =chain_rgroup[-1][:, :, :model.n_dims]
                h_rgroup =chain_rgroup[-1][:, :, model.n_dims:]

                x_rgroup += mean


                node_mask_r =  sample_atom_mask -sample_pocket_mask


                # 保存完整分子
                pred_names_r = [f'{uuid}/{i}_full' for uuid in uuids]
                vis.save_xyz_file_fa(output_dir, h_rgroup, x_rgroup,node_mask_r, pred_names_r)



                all_valid = True
                # 转换为 SDF 文件
                for j in range(len(pred_names_r)):
                    out_xyz = f'{output_dir}/{pred_names_r[j]}_.xyz'
                    opt_xyz = f'{output_dir}/{pred_names_r[j]}_opt.xyz'
                    out_sdf = f'{output_dir}/{pred_names_r[j]}_.sdf'
                    obabel_path = '/home/cht/anaconda3/envs/BF/bin/obabel'  # 换成你的 obabel 路径

                    # 第一步：优化 xyz 文件，输出优化后的 xyz
                    result1 = subprocess.run(
                        f'{obabel_path} {out_xyz} -O {opt_xyz} --minimize --ff UFF',
                        shell=True, capture_output=True, text=True
                    )

                    # 第二步：将优化后的 xyz 转换为 sdf
                    result2 = subprocess.run(
                        f'{obabel_path} {opt_xyz} -O {out_sdf}',
                        shell=True, capture_output=True, text=True
                    )
                    if result2.returncode == 0 and os.path.exists(out_sdf) and os.path.getsize(out_sdf) > 0:
                        if is_valid_molecule(out_sdf):
                            sa_score = get_sa_score_from_sdf(out_sdf)
                            qed =calculate_qed_from_single_sdf(out_sdf)
                            if sa_score is not None and qed is not None and (
                                    (qed > 0.6 and sa_score > 0.4) or
                                    (sa_score > 0.6 and qed > 0.4) or
                                    (qed > 0.7 and sa_score > 0.3) or
                                    (sa_score > 0.7 and qed > 0.3) or
                                    (qed > 0.5 and sa_score > 0.5) or
                                    (sa_score > 0.8 and qed > 0.2) or (sa_score > 0.2 and qed > 0.8)
                            ):

                                print(f"Generated {out_sdf} successfully, valid and SA score = {sa_score:.2f}")
                            else:
                                print(
                                    f"Generated molecule {out_sdf} has low SA score ({sa_score:.2f}), deleting and retrying...")
                                os.remove(out_xyz)
                                os.remove(opt_xyz)
                                os.remove(out_sdf)
                                all_valid = False
                        else:
                            print(f"Molecule {out_sdf} is invalid, deleting and retrying...")
                            os.remove(out_xyz)
                            os.remove(opt_xyz)
                            os.remove(out_sdf)

                            all_valid = False
                    else:
                        print(f"Warning: Failed to convert {out_xyz} to SDF")
                        if os.path.exists(out_xyz):
                            os.remove(out_xyz)
                        if os.path.exists(opt_xyz):  # <-- 加这一行
                            os.remove(opt_xyz)
                        all_valid = False
                if all_valid:
                    break  # 所有分子都合理，跳出重试循环
                else:
                    retries += 1
                    if retries >=1000:
                        print(
                            f"Failed to generate valid molecule after {1000} retries for sample {i}, uuids {uuids}")
                        break

            except Exception as e:
                print(f"Error in sampling molecule {i} for uuids {uuids}: {e}")
                continue


def main():
    args1 = argparse.Namespace(
        checkpoint='/home/cht/DiffDec-master/DDPM/logs/crossdock_exp_2025_05_28__16_31_19/checkpoints/fold_1_best_scaffold_1122.pt',
        samples='./output_samples_cheat',
        device='cuda',
        n_samples=100
    )

    # 设置输出目录
    experiment_name = args1.checkpoint.split('/')[-1].replace('.pt', '')
    output_dir = os.path.join(args1.samples, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    args = p.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    args.dataset = 'crossdock'
    config_path = '/home/cht/DiffDec-master/DDPM/configs/single_full.yml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config.diffusion_steps = args.diffusion_steps
    config.diffusion_noise_schedule = args.diffusion_noise_schedule
    config.diffusion_noise_precision = args.diffusion_noise_precision
    config.diffusion_loss_type = args.diffusion_loss_type

    model = DDPM(
        data_path=config.data_path,
        train_data_prefix=config.train_data_prefix,
        val_data_prefix=config.val_data_prefix,
        in_node_nf=config.in_node_nf,
        n_dims=3,
        context_node_nf=config.context_node_nf,
        hidden_nf=config.nf,
        activation=config.activation,
        n_layers=config.n_layers,
        attention=config.attention,
        tanh=config.tanh,
        norm_constant=config.norm_constant,
        inv_sublayers=config.inv_sublayers,
        sin_embedding=config.sin_embedding,
        normalization_factor=config.normalization_factor,
        aggregation_method=config.aggregation_method,
        diffusion_steps=config.diffusion_steps,
        diffusion_noise_schedule=config.diffusion_noise_schedule,
        diffusion_noise_precision=config.diffusion_noise_precision,
        diffusion_loss_type=config.diffusion_loss_type,
        normalize_factors=config.normalize_factors,
        include_charges=config.include_charges,
        lr=config.lr,
        batch_size=config.batch_size,
        torch_device=args1.device,
        model=config.model,
        test_epochs=config.test_epochs,
        n_stability_samples=config.n_stability_samples,
        normalization=config.normalization,
        log_iterations=config.log_iterations,
        samples_dir=config.samples_dir,
        data_augmentation=config.data_augmentation,
        center_of_mass=config.center_of_mass,
        inpainting=False,
        anchors_context=True,
    )

    # 手动加载检查点
    checkpoint = torch.load(args1.checkpoint, map_location=args1.device)

    model.load_state_dict(checkpoint['model'])  # 使用 'model' 键加载状态字典
    model = model.eval().to(args1.device)


    # 后续代码保持不变
    val_dataset = CrossDockDataset(
        data_path='/home/cht/DiffDec-master/data/data2',  # 替换为实际数据路径
        prefix='crossdocksingle_test.full',                    # 数据集前缀，根据你的需求调整
        device=args1.device
    )
    dataloader = get_dataloader(val_dataset, batch_size=1, collate_fn=collate)
    print(f'Dataloader contains {len(dataloader)} batches')

    target_batch_idx = 36 # 替换为您想处理的样本索引，例如 0 表示第一个样本

    # 获取指定索引的样本
    data_iterator = iter(dataloader)
    for i in range(target_batch_idx + 1):
        try:
            data = next(data_iterator)
        except StopIteration:
            print(f"错误：目标索引 {target_batch_idx} 超出 DataLoader 长度 {len(dataloader)}")
            return

    # 处理单个样本
    time_start = time.time()

    uuids = []
    true_names = []
    frag_names = []
    scaf_names = []
    pock_names = []
    for uuid in data['uuid']:
        uuid = str(uuid)
        uuids.append(uuid)
        true_names.append(f'{uuid}/true')
        frag_names.append(f'{uuid}/frag')
        scaf_names.append(f'{uuid}/scaf')
        pock_names.append(f'{uuid}/pock')
        os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

    # 检查是否已生成
    generated, starting_point = check_if_generated(output_dir, uuids, args1.n_samples)
    if generated:
        print(f'样本 batch={target_batch_idx}, max_uuid={max(uuids)} 已生成，跳过')
        return
    if starting_point > 0:
        print(f'为 batch={target_batch_idx} 生成剩余 {args1.n_samples - starting_point} 个样本')

    # 提取数据
    h, x, node_mask, scaffold_mask, fragment_mask = data['one_hot'], data['positions'], data['atom_mask'], data[
        'scaffold_mask'], data['fragment_mask']
    node_mask = data['atom_mask'] - data['pocket_mask']
    pock_mask = data['pocket_mask']

    # 保存初始文件
    vis.save_xyz_file_fa(output_dir, h, x, pock_mask, pock_names)
    vis.save_xyz_file_fa(output_dir, h, x, fragment_mask, frag_names)
    vis.save_xyz_file_fa(output_dir, h, x, node_mask, true_names)
    vis.save_xyz_file_fa(output_dir, h, x, scaffold_mask, scaf_names)

    # 执行采样
    two_stage_sampling(model, data, output_dir, uuids, args1.n_samples, starting_point)

    time_end = time.time()
    print(f'样本 {target_batch_idx} 处理完成，总耗时: {time_end - time_start:.2f} 秒')


if __name__ == "__main__":
    main()