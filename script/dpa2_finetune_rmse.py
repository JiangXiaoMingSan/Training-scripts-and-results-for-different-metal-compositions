import dpdata
import numpy as np
import argparse
from pathlib import Path

class DPPTPredict:
    def load_systems(self, data_path: str):
        systems = []
        data_path = Path(data_path)

        for set_path in data_path.rglob('set.000'):
            parent = set_path.parent
            try:
                system = dpdata.LabeledSystem(parent, fmt="deepmd/npy")
                systems.append((parent.name, system))
            except Exception as e:
                print(f"Could not load system from {parent}: {e}")
        return systems

    def compute_errors(self, predicted, actual):
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        return mae, rmse

    def compare_predictions(self, predicted_data_path: str, actual_data_path: str):
        predicted_systems = self.load_systems(predicted_data_path)
        actual_systems = self.load_systems(actual_data_path)

        energy_mae_list = []
        energy_rmse_list = []
        energy_mae_per_atom_list = []
        energy_rmse_per_atom_list = []
        force_mae_list = []
        force_rmse_list = []
        total_test_data = 0
        valid_systems = 0

        for (name, predicted_sys), (_, actual_sys) in zip(predicted_systems, actual_systems):
            try:
                nframes = predicted_sys["energies"].shape[0]
                natoms = predicted_sys["coords"].shape[1]

                total_test_data += nframes

                predicted_energies = predicted_sys["energies"]
                actual_energies = actual_sys["energies"]
                e_mae, e_rmse = self.compute_errors(predicted_energies, actual_energies)
                e_mae_per_atom = e_mae / natoms
                e_rmse_per_atom = e_rmse / natoms

                predicted_forces = predicted_sys["forces"].reshape(-1)
                actual_forces = actual_sys["forces"].reshape(-1)
                f_mae, f_rmse = self.compute_errors(predicted_forces, actual_forces)

                print(f"# System: {name}")
                print(f"  Number of test data : {nframes}")
                print(f"  Energy MAE          : {e_mae:.6e} eV")
                print(f"  Energy RMSE         : {e_rmse:.6e} eV")
                print(f"  Energy MAE/Natoms   : {e_mae_per_atom:.6e} eV")
                print(f"  Energy RMSE/Natoms  : {e_rmse_per_atom:.6e} eV")
                print(f"  Force  MAE          : {f_mae:.6e} eV/A")
                print(f"  Force  RMSE         : {f_rmse:.6e} eV/A")
                print("-" * 50)

                energy_mae_list.append(e_mae)
                energy_rmse_list.append(e_rmse)
                energy_mae_per_atom_list.append(e_mae_per_atom)
                energy_rmse_per_atom_list.append(e_rmse_per_atom)
                force_mae_list.append(f_mae)
                force_rmse_list.append(f_rmse)

                valid_systems += 1
            except Exception as e:
                print(f"Skipping system {name} due to error: {e}")

        if valid_systems == 0:
            print("No valid systems were loaded. Exiting.")
            return

        overall_energy_mae = np.mean(energy_mae_list)
        overall_energy_rmse = np.mean(energy_rmse_list)
        overall_energy_mae_per_atom = np.mean(energy_mae_per_atom_list)
        overall_energy_rmse_per_atom = np.mean(energy_rmse_per_atom_list)
        overall_force_mae = np.mean(force_mae_list)
        overall_force_rmse = np.mean(force_rmse_list)

        print("# ---------- Weighted (Arithmetic) Average of Errors -----------")
        print(f"Number of systems           : {valid_systems}")
        print(f"Total Number of test data   : {total_test_data}")
        print(f"Overall Energy MAE          : {overall_energy_mae:.6e} eV")
        print(f"Overall Energy RMSE         : {overall_energy_rmse:.6e} eV")
        print(f"Overall Energy MAE/Natoms   : {overall_energy_mae_per_atom:.6e} eV")
        print(f"Overall Energy RMSE/Natoms  : {overall_energy_rmse_per_atom:.6e} eV")
        print(f"Overall Force MAE           : {overall_force_mae:.6e} eV/A")
        print(f"Overall Force RMSE          : {overall_force_rmse:.6e} eV/A")
        print("# -------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("actual_path", help="Path to the actual data (e.g., valid)")
    parser.add_argument("predicted_path", help="Path to the predicted data (e.g., valid_predict)")
    args = parser.parse_args()

    predictor = DPPTPredict()
    predictor.compare_predictions(args.predicted_path, args.actual_path)

