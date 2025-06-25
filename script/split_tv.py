import dpdata
from IPython import embed
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
import os
import shutil
import numpy as np


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default='.',
        help="path to data",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default='.',
        help="output dir",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=0.2,
        help="rate for valid",
    )
    parser.add_argument(
        "--mixed-type", action="store_true", default=False, help="Process mixed type format."
    )
    parsed_args = parser.parse_args(args=args)
    return parsed_args


if __name__ == '__main__':
    args = parse_args()
    dict_args = vars(args)
    datapath = dict_args['path']
    rate = dict_args['rate']
    outdir = dict_args['out']
    os.makedirs(outdir, exist_ok=True)
    mixed_type = dict_args['mixed_type']
    sys_src = []
    sys_out_train = []
    sys_out_valid = []
    train_sys = []
    valid_sys = []
    for root, dirs, files in os.walk(datapath):
        if "type.raw" in files:
            sys_src.append(root)
            relative_path = os.path.relpath(root, datapath)
            train_sys_dir = os.path.join(outdir, relative_path, "train")
            os.makedirs(train_sys_dir, exist_ok=True)
            sys_out_train.append(train_sys_dir)
            valid_sys_dir = os.path.join(outdir, relative_path, "valid")
            os.makedirs(valid_sys_dir, exist_ok=True)
            sys_out_valid.append(valid_sys_dir)

    for i in tqdm(range(len(sys_src))):
        if not mixed_type:
            sys = dpdata.LabeledSystem(sys_src[i], fmt='deepmd/npy')
            sys.shuffle()
            sys_num = len(sys)
            valid_num = int(sys_num * rate)
            if valid_num>0:
                train = sys[:-valid_num]
                valid = sys[-valid_num:]
            else:
                train = sys
            train.to_deepmd_npy(sys_out_train[i])
            train_sys.append(os.path.abspath(sys_out_train[i])+'\n')
            if valid_num > 0:
                valid.to_deepmd_npy(sys_out_valid[i])
                valid_sys.append(os.path.abspath(sys_out_valid[i])+'\n')
            else:
                shutil.rmtree(sys_out_valid[i])
            print(f"split {sys_src[i]} to train[{sys_num-valid_num}]: {sys_out_train[i]} and valid[{valid_num}]: {sys_out_valid[i]}")
        else:
            sys = dpdata.LabeledSystem(sys_src[i], fmt='deepmd/npy')
            type_map = open(os.path.join(sys_src[i], "type_map.raw")).read().split()
            sys.data['atom_names'] = type_map
            sys_num = len(sys)
            index = np.arange(sys_num)
            np.random.shuffle(index)
            valid_num = int(sys_num * rate)
            if valid_num > 0:
                train = sys[index[:-valid_num]]
                valid = sys[index[-valid_num:]]
            else:
                train = sys
            train.to_deepmd_npy(sys_out_train[i])
            train_sys.append(os.path.abspath(sys_out_train[i]) + '\n')
            if valid_num > 0:
                valid.to_deepmd_npy(sys_out_valid[i])
                valid_sys.append(os.path.abspath(sys_out_valid[i]) + '\n')
            else:
                shutil.rmtree(sys_out_valid[i])
            print(
                f"split {sys_src[i]} to train[{sys_num - valid_num}]: {sys_out_train[i]} and valid[{valid_num}]: {sys_out_valid[i]}")
    os.makedirs(os.path.join(outdir, "sys_dir"), exist_ok=True)
    f_train = open(os.path.join(outdir,  "sys_dir", "train.txt"), "w")
    f_train.writelines(train_sys)
    f_valid = open(os.path.join(outdir, "sys_dir", "valid.txt"), "w")
    f_valid.writelines(valid_sys)
