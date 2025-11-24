# YuelBond: Multimodal Bonds Reconstruction Towards Generative Molecular Design

Generative models such as diffusion-based approaches have transformed de novo drug design by enabling rapid generation of novel molecular structures in both 2D and 3D formats. However, accurate reconstruction of chemical bonds, especially from distorted geometries produced by generative models, remains a critical challenge. Here, we present YuelBond, a multimodal graph neural network framework for robust bonds reconstruction across three key scenarios: (i) recovery of bonds from accurate 3D atomic coordinates, (ii) reconstruction of chemically valid bonds in crude de novo generated compounds (CDGs) with perturbed geometries, and (iii) reassignment of bond orders in 2D topological graphs. YuelBond outperforms traditional rule-based methods such as RDKit, achieving 98.4% F1-score on standard 3D structures and maintaining strong performance (92.7% F1-score) on distorted CDGs, even when RDKit fails in most cases. Our results demonstrate that YuelBond enables accurate and reliable bond reconstruction from imperfect molecular data, bridging a critical gap in generative drug discovery pipelines.

## Environment Setup

installing all the necessary packages:

```shell
rdkit
biopython
pytorch
pytorch-lightning
scipy
scikit-learn
tqdm
wandb
```

## Usage

### Download Datasets

```shell
mkdir -p datasets
# Sanitized 3D 
wget https://zenodo.org/records/15353365/files/geom_train.pt -O datasets/geom_train.pt
# Sanitized CDG
wget https://zenodo.org/records/15353365/files/geom_sanitized_train_noise_0_2.pt -O datasets/geom_sanitized_train_noise_0_2.pt
# Sanitized 2D
wget https://zenodo.org/records/15353365/files/geom_train_bonds.pt -O datasets/geom_train_bonds.pt
# Kekulized 3D
wget https://zenodo.org/records/15353365/files/geom_kekulized_train.pt -O datasets/geom_kekulized_train.pt
# Kekulized CDG
wget https://zenodo.org/records/15353365/files/geom_kekulized_train_noise_0_2.pt -O datasets/geom_kekulized_train_noise_0_2.pt
# Kekulized 2D
wget https://zenodo.org/records/15353365/files/geom_kekulized_train_bonds.pt -O datasets/geom_kekulized_train_bonds.pt
```

#### Download Models

```shell
mkdir -p models
# Sanitized 3D
wget https://zenodo.org/records/15353365/files/geom_3d.ckpt -O models/geom_3d.ckpt
# Sanitized CDG
wget https://zenodo.org/records/15353365/files/geom_cdg.ckpt -O models/geom_cdg.ckpt
# Sanitized 2D
wget https://zenodo.org/records/15353365/files/geom_2d.ckpt -O models/geom_2d.ckpt
# Kekulized 3D
wget https://zenodo.org/records/15353365/files/geom_kekulized_3d.ckpt -O models/geom_kekulized_3d.ckpt
# Kekulized CDG
wget https://zenodo.org/records/15353365/files/geom_kekulized_cdg.ckpt -O models/geom_kekulized_cdg.ckpt
# Kekulized 2D
wget https://zenodo.org/records/15353365/files/geom_kekulized_2d.ckpt -O models/geom_kekulized_2d.ckpt
```

#### Predicting Bonds

Predicting bonds from 3D structures:
```shell
# test.xyz is the 3D structure to predict bonds from
# test.sdf is the output SDF file with predicted bonds
python yuel_bond.py test.xyz test.sdf --model models/geom_3d.ckpt
```

Predicting bonds from CDGs:
```shell
python yuel_bond.py test.xyz test.sdf --model models/geom_cdg.ckpt
```

Predicting bonds from 2D structures:
```shell
# test.sdf is the input SDF file with 2D structure
# test_2d.sdf is the output SDF file with predicted bonds
python yuel_bond.py test.sdf test_2d.sdf --model models/geom_2d.ckpt --mode 2d
```

Predicting bonds from kekulized 3D structures:
```shell
# test.xyz is the 3D structure to predict bonds from
# test.sdf is the output SDF file with predicted kekulized bonds
python yuel_bond.py test.xyz test.sdf --model models/geom_kekulized_3d.ckpt
```

Predicting bonds from kekulized CDGs:
```shell
# test.xyz is the 3D structure to predict bonds from
# test.sdf is the output SDF file with predicted kekulized bonds
python yuel_bond.py test.xyz test.sdf --model models/geom_kekulized_cdg.ckpt
```

Predicting bonds from kekulized 2D structures:
```shell
# test.sdf is the input SDF file with 2D structure
# test_2d.sdf is the output SDF file with predicted kekulized bonds
python yuel_bond.py test.sdf test_2d.sdf --model models/geom_kekulized_2d.ckpt --mode 2d
```

### Training

Training Sanitized 3D:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom.yml
```

Training Sanitized CDG:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom_sanitized_noise_0_2.yml
```

Training Sanitized 2D:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom_bonds.yml
```

Training Kekulized 3D:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom_kekulized.yml
```

Training Kekulized CDG:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom_kekulized_noise_0_2.yml
```

Training Kekulized 2D:
```shell
python -W ignore train_yuel_bond.py --config configs/train_geom_kekulized_bonds.yml
```

# Contact

If you have any questions, please contact me at jianopt@gmail.com
