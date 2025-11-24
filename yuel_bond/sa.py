import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from sascorer import calculateScore

base_path = "/home/cht/dokhlab-yuel_bond/test"

def compute_scores_for_sdf_folder(sdf_folder):
    sa_scores = []
    qed_scores = []
    for filename in os.listdir(sdf_folder):
        if filename.endswith(".sdf") and "out" not in filename:
            sdf_path = os.path.join(sdf_folder, filename)
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)
            for mol in suppl:
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        sa = calculateScore(mol)
                        sa_norm = round((10 - sa) / 9, 4)
                        qed = round(QED.qed(mol), 4)

                        sa_scores.append(sa_norm)
                        qed_scores.append(qed)
                    except Exception as e:
                        print(f"⚠️ Skipped molecule in {filename} due to error: {e}")
    return sa_scores, qed_scores

def main():
    all_sa_scores = []
    all_qed_scores = []

    with open("average_sa_scores.txt", "w") as f:
        f.write("Folder\tMolecule_Count\tAverage_SA\tStd_SA\tAverage_QED\tStd_QED\n")

        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                sa_scores, qed_scores = compute_scores_for_sdf_folder(folder_path)
                if sa_scores and qed_scores:
                    count = len(sa_scores)
                    avg_sa = np.mean(sa_scores)
                    std_sa = np.std(sa_scores)
                    avg_qed = np.mean(qed_scores)
                    std_qed = np.std(qed_scores)

                    all_sa_scores.extend(sa_scores)
                    all_qed_scores.extend(qed_scores)

                    print(f"{folder}: {count} molecules | Avg SA = {avg_sa:.4f} | Std SA = {std_sa:.4f} | Avg QED = {avg_qed:.4f} | Std QED = {std_qed:.4f}")
                    f.write(f"{folder}\t{count}\t{avg_sa:.4f}\t{std_sa:.4f}\t{avg_qed:.4f}\t{std_qed:.4f}\n")
                else:
                    print(f"{folder}: No valid molecules found.")
                    f.write(f"{folder}\t0\tN/A\tN/A\tN/A\tN/A\n")

        # === Overall statistics for all folders ===
        if all_sa_scores and all_qed_scores:
            total_count = len(all_sa_scores)
            total_avg_sa = np.mean(all_sa_scores)
            total_std_sa = np.std(all_sa_scores)
            total_avg_qed = np.mean(all_qed_scores)
            total_std_qed = np.std(all_qed_scores)

            count_qed_above_06 = sum(1 for q in all_qed_scores if q > 0.6)
            proportion_qed_above_06 = count_qed_above_06 / total_count

            print("\n=== Overall Statistics ===")
            print(f"Total molecules: {total_count}")
            print(f"Overall Average SA = {total_avg_sa:.4f}, Std = {total_std_sa:.4f}")
            print(f"Overall Average QED = {total_avg_qed:.4f}, Std = {total_std_qed:.4f}")
            print(f"QED > 0.6 count: {count_qed_above_06}, Proportion: {proportion_qed_above_06:.4%}")

            f.write("\n=== Overall Statistics ===\n")
            f.write(f"Total_molecules\t{total_count}\n")
            f.write(f"Overall_Average_SA\t{total_avg_sa:.4f}\n")
            f.write(f"Overall_Std_SA\t{total_std_sa:.4f}\n")
            f.write(f"Overall_Average_QED\t{total_avg_qed:.4f}\n")
            f.write(f"Overall_Std_QED\t{total_std_qed:.4f}\n")
            f.write(f"QED>0.6_Count\t{count_qed_above_06}\n")
            f.write(f"QED>0.6_Proportion\t{proportion_qed_above_06:.4%}\n")
        else:
            print("\nNo valid molecules found in any folder.")
            f.write("\nNo valid molecules found in any folder.\n")

if __name__ == "__main__":
    main()
