import os
import subprocess

# 设置目标目录
input_dir = '/home/cht/dokhlab-yuel_bond/test'
output_dir ='/home/cht/dokhlab-yuel_bond/test/1'
# 获取所有xyz文件
xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]

# 遍历每个文件
for xyz_file in xyz_files:
    # 构建输入和输出路径
    input_path = os.path.join(input_dir, xyz_file)
    output_path = os.path.join(output_dir, f'{os.path.splitext(xyz_file)[0]}.sdf')

    # 构建命令
    command = ['python3', 'yuel_bond.py', input_path, output_path, '--model', 'models/geom_3d.ckpt']

    # 执行命令
    subprocess.run(command)

print("批量处理完成！")
