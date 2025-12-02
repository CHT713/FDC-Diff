# FDC-Diff：Fragment-based Dual Conditional Diffusion Framework for Molecular Generation
<p align="center">
<img src="figure/model.png" alt="architecture"/>
</p>

## 🧩 Dependencies

This project requires a conda environment for dependency management. Please use the provided environment file to install all required packages.

### 🔧 Setup via Conda

```bash
# Clone the repository (if not done already)
git clone https://github.com/CHT713/FDC-Diff.git
cd FDC-Diff

# Create the environment from the YAML file
conda env create -f FC.yaml(If you encounter download failures, we recommend that you manually download each package that failed individually.)

# Activate the environment
conda activate FC
 ```

## 📦 Dataset Preparation
We use the CrossDocked dataset and the reaction-based slicing method from LibINVENT to construct datasets.To prepare the datasets from scratch, follow the steps below:

1.Download the dataset archive crossdocked_pocket10.tar.gz and the split file split_by_name.pt.(https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM)

 You can also find the original CrossDocked dataset at:https://bits.csb.pitt.edu/files/crossdock2020/
 
2.Extract the TAR archive using the command:
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```

3.Split raw PL‑complexes and convert SDF to SMILES:
```bash
python split_and_convert.py
```

4.Use the reaction-based slicing method in [LibINVENT](https://github.com/MolecularAI/Lib-INVENT-dataset)  to slice the molecules into scaffolds and R-groups in Lib-INVENT-dataset and replace and replace example_configurations/supporting_files/filter_conditions.json in Lib-INVENT-dataset with filter_conditions.json in this directory of data.

5.Process datasets：Switch to the data/single directory，Save the results obtained from libinvent, then modify the train_sliced_file, test_sliced_file (the save path for libinvent results), processed_train_file, and processed_test_file (the save path for the final processed data structure) in the data/single/process_and_prepare file.
```bash
python -W ignore process_and_prepare.py
```

## 📦 Training
```bash
python train.py --config configs/single_full.yml
```

## 📦 Sampling
Switch to the models folder under the yuel_bond directory, and run the following commands to download the bond prediction model weights:
```bash
wget https://zenodo.org/records/15353365/files/geom_3d.ckpt -O models/geom_3d.ckpt
wget https://zenodo.org/records/15353365/files/geom_cdg.ckpt -O models/geom_cdg.ckpt
```

```bash
python sample.py
```
## Other datasets
If you want to train and test other datasets on our model, first, you need to understand how to use LibINVENT to split ligand data into scaffolds and R-groups. Afterward, modify the process_and_prepare.py script to accommodate the input and output requirements of your dataset. Finally, if you still encounter issues, you may need to modify the dataset.py file in the DDPM directory to ensure compatibility with the input of your dataset.
