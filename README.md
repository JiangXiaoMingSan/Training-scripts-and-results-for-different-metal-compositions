The large-scale model used in DPA2 was obtained from the AISquare model repository, available at:


https://www.aissquare.com/models/detail?pageType=models&name=DPA-2.3.1-v3.0.0rc0&id=287

The training was performed with the following command:


dp --pt train input_finetune.json --finetune DPA2_medium_28_10M_rc0_AIS.pt --model-branch Domains_Alloy

An example of the folder naming convention is as follows:


2alloy_train refers to the model trained using all binary alloy training data.


2alloy_train_valid refers to the model trained using both the full binary alloy training and test datasets.


2alloy_train_valid20 indicates that the model was trained using the full binary alloy training set and 20% of the test set.


The test set was randomly sampled using the script/split_tv.py script.



