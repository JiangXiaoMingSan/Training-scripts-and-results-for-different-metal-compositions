import dpdata
import numpy as np
import os
from deepmd.infer import DeepPot
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple

class DPPTPredict:
    def load_model(self, model: Path):
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        e, f, v = self.dp.eval(coord, cell, atype)
        return e.reshape([-1]), f, v.reshape([-1,3,3])

    def predict(self, input_path="input", output_path="output"):
        type_map = self.dp.get_type_map()
        for f in tqdm(list(Path(input_path).rglob("type.raw"))):
            sys = f.parent
            print(sys)
            d = dpdata.MultiSystems()
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)
            for k in d:
                anames = k["atom_names"]
                ori_atype = k["atom_types"]
                atype = np.array([type_map.index(anames[j]) for j in ori_atype])
                e, f, v = self.evaluate(k["coords"], k["cells"], atype)
                k.data["energies"] = e
                k.data["forces"] = f
                k.data["virials"] = v
            # For configurations in DP-Gen2 only accept 1-level dir
            out_dir = os.path.join(output_path, str(sys.relative_to(input_path)).replace("/", "_"))
            if len(d) == 1:
                d[0].to_deepmd_npy_mixed(out_dir)
            else:
                # The multisystem is loaded from one dir, thus we can safely keep one dir
                d.to_deepmd_npy_mixed(out_dir + ".tmp")
                fs = os.listdir(out_dir + ".tmp")
                assert len(fs) == 1
                os.rename(os.path.join(out_dir + ".tmp", fs[0]), out_dir)
                os.rmdir(out_dir + ".tmp")

d = DPPTPredict()
d.load_model('teacher_model.pt')
d.predict('single', 'single_predict')
d.predict('2alloy', '2alloy_predict')
d.predict('3alloy', '3alloy_predict')
d.predict('4alloy', '4alloy_predict')
d.predict('5alloy', '5alloy_predict')
