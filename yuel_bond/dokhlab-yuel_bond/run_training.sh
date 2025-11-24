module load miniconda/3
conda activate torch

python -W ignore train_yuel_bond.py --config configs/train_geom_kekulized_noise_0_2.yml
