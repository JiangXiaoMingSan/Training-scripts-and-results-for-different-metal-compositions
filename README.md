**README**

---

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) \[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]

## 🔗 Model Source

The base model is obtained from the AISquare Model Repository:

* **DPA2_medium_28_10M_rc0_AIS.pt**: [View on AISquare](https://www.aissquare.com/models/detail?pageType=models&name=DPA-2.3.1-v3.0.0rc0&id=287)

## 🚀 Training

Fine-tune the model with your dataset using the following command:

```bash
dp --pt train input_finetune.json \
   --finetune DPA2_medium_28_10M_rc0_AIS.pt \
   --model-branch Domains_Alloy
```

* `input_finetune.json`: JSON config for fine-tuning
* `DPA2_medium_28_10M_rc0_AIS.pt`: AIS model
* `--model-branch`: Branch name to use (e.g., `Domains_Alloy`)

## 📂 Folder Naming Convention

| Folder Name            | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| `2alloy_train`         | Trained on **all** binary alloy training data.                            |
| `2alloy_train_valid`   | Trained on binary alloy training **+ full test** datasets.                |
| `2alloy_train_valid20` | Trained on training dataset + **20%** of the test dataset (random split). |

> The test set split is handled by `script/split_tv.py` for reproducibility.

## 🔍 Prediction & Testing

Since `dp test` cannot directly evaluate fine-tuned models, use the `DPPTPredict.py` script.

## 📊 Evaluation Script

Use `dpa2_finetune_rmse.py` to compute RMSE between predictions and reference data:

```bash
python dpa2_finetune_rmse.py <reference_dir> <predict_dir>
```

* `<reference_dir>`: Directory of original data
* `<predict_dir>`: Directory of predicted results

The script prints per-system and global RMSE metrics.

## 💡 Example Usage

```bash
# 1. Fine-tune the model
dp --pt train input_finetune.json \
   --finetune DPA2_medium_28_10M_rc0_AIS.pt \
   --model-branch Domains_Alloy

# 2. Predict with fine-tuned model
python DPPTPredict.py 

# 3. Evaluate RMSE
python dpa2_finetune_rmse.py data/binary_alloys/train predictions/2alloy_train
```

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve scripts, add new datasets, or incorporate additional evaluation metrics.

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
