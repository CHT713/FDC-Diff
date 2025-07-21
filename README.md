# FDC-Diff：Fragment-based Dual Conditional Diffusion Framework for Molecular Generation

## 🧩 Dependencies

This project requires a conda environment for dependency management.  
Please use the provided environment file to install all required packages.

### 🔧 Setup via Conda

```bash
# Clone the repository (if not done already)
git clone https://github.com/CHT713/FDC-Diff.git
cd FDC-Diff

# Create the environment from the YAML file
conda env create -f FC.yml

# Activate the environment
conda activate FC
 ```

## 📦 Dataset Preparation
We use the CrossDocked dataset and the reaction-based slicing method from LibINVENT to construct single and multi R-group datasets.To prepare the datasets from scratch, follow the steps below:

1.Download the dataset archive crossdocked_pocket10.tar.gz and the split file split_by_name.pt.
  You can also find the original CrossDocked dataset at:https://bits.csb.pitt.edu/files/crossdock2020/
2.Extract the TAR archive using the command:
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```
3.Split raw PL‑complexes and convert SDF to SMILES:
```bash
python split_and_convert.py
```
4.Use the reaction-based slicing method in [LibINVENT](https://github.com/MolecularAI/Lib-INVENT-dataset)  to slice the molecules into scaffolds and R-groups in Lib-INVENT-dataset and replace
5.Process datasets
```bash
python -W ignore process_and_prepare.py
```

## 📦 Training
```bash
python train.py --config configs/single_full.yml
```

## 📦 Sampling
```bash
python sample.py -W ignore --checkpoint ckpt/best.ckpt \
                 --samples sample_mols \
                 --data data/single \
                 --prefix crossdocksingle_test_full \
                 --n_samples 100 \
                 --device cuda:0
```
