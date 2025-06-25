import dpdata
from pathlib import Path
import argparse

def count_total_test_data(data_path: str):
    total_frames = 0
    for subdir in sorted(Path(data_path).iterdir()):
        if subdir.is_dir():
            try:
                system = dpdata.LabeledSystem(subdir, fmt="deepmd/npy")
                nframes = system["energies"].shape[0]
                print(f"{subdir.name}: Number of test data = {nframes}")
                total_frames += nframes
            except Exception as e:
                print(f"Could not load system from {subdir}: {e}")
    print("--------------------------------------------------")
    print(f"Total Number of test data = {total_frames}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the directory containing test systems")
    args = parser.parse_args()

    count_total_test_data(args.data_path)

