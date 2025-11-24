# geom_3d
# python yuel_bond.py datasets/geom_sanitized_test ./tests --model models/geom_bs16_date18-04_time16-47-04.989651/last.ckpt --dataset --batch_size 4

# geom_cdg
# python yuel_bond.py datasets/geom_sanitized_test_noise_0_2 ./tests --model models/geom_sanitized_noise_0_2_bs16_date25-04_time11-01-59.742623/last.ckpt --dataset --batch_size 4

# geom_2d
# python yuel_bond.py datasets/geom_sanitized_test ./tests --model models/geom_bonds_bs16_date21-04_time00-38-57.668084/last.ckpt --dataset --batch_size 4 --has_bonds 

# geom_kekulized_3d
# python yuel_bond.py datasets/geom_kekulized_test ./tests --model models/geom_kekulized_bs16_date22-04_time18-00-22.108234/last.ckpt --dataset --batch_size 4

# geom_kekulized_cdg
python yuel_bond.py datasets/geom_kekulized_test_noise_0_2 ./tests --model models/geom_kekulized_noise_0_2_bs16_date03-05_time13-43-59.050578/last.ckpt --dataset --batch_size 4

# geom_kekulized_2d
# python yuel_bond.py datasets/geom_kekulized_test ./tests --model models/geom_kekulized_bonds_bs16_date22-04_time18-12-42.531500/last.ckpt --dataset --batch_size 4 --has_bonds

