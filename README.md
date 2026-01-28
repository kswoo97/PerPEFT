# PerPEFT

This repository is an official implementation of PerPEFT, a personalized parameter-efficient fine-tuning method for multimodal recommendation.

### Paper 

- Title: ***PerPEFT: Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendations***
- Authors: Sunwoo Kim, Hyunjin Hwang, Kijung Shin
- Venues: The Web Conference 2026 (WWW 2026)
- Affiliation: KAIST AI

----

### Datasets

In this work, we use the four datasets from https://amazon-reviews-2023.github.io/.

| Name | \# Users | \# Items |
|:------|-------:|:----:|
| Sports & Outdoors   | 25,363  | 15,701 |
| Toys & Games   | 19,026   | 14,718 |
| Beauty & Personal Care | 45,490  | 31,151 |
| Arts, Crafts & Sewing    | 24,511 | 18,884 |

The entire dataset, including (1) user-item interactions, (2) item images, and (3) item titles, is presented in the link below:
- **Link**: https://www.dropbox.com/scl/fo/olpz13hyfcdn5jg6a4tzy/AN1Na5w_ySyO4nnYFyYnRpE?rlkey=5i6iyloeq9fpa48tz29ztmaqj&st=ibnulph8&dl=0
- **Description**: Refer to the ```README.txt``` file within the link.

Once you download the datasets in the above link, all files will be downloaded in ```mmrec_datasets.zip``` file.
Then, locate the zip file in your directory, and run the file below to make the correct dataset directories:
```
python3 unzip_data.py
```

----

### Key packages

Our implementation is conducted upon the following key packages:

```
python == 3.8.20
pytorch == 2.2.0+cu122
numpy == 1.24.3
peft == 0.13.2
transformers == 4.45.2
```

Note that several other versions are also compatible.
Feel free to **leave any issues** caused by the package error.

------


### How to run the code?

In this work, we use three PEFT modules: (1) LoRA, (2) (IA)3, and (3) IISAN.

- For LoRA (standard PEFT module), one can use the code below:
```
python3 perpeft_lora_ia3.py --dataset sports_outdoors --device cuda:0 --peft_type lora --wdecay 1e-4 --lr 1e-4
```
- For (IA)3 (the strongest PEFT module in our setting), one can use the code below:
```
python3 perpeft_lora_ia3.py --dataset sports_outdoors --device cuda:0 --peft_type ia3 --wdecay 1e-4 --lr 1e-4
```
- For IISAN (the fastest PEFT module in our setting), one can use the code below:
```
python3 perpeft_iisan.py --dataset sports_outdoors --device cuda:0 --wdecay 1e-4 --lr 1e-4
```

Description for each hyperparameter is as follows:
- ```--dataset``` indicates the target dataset one aims to use. One can choose: (1) sports_outdoors, (2) toys_games, (3) beauty_care, and (4) arts.
- ```--device``` indicates the target GPU device one aims to use.
- ```--wdecay``` indicates the coefficient for the weight normalization of the (1) PEFT modules and (2) backbone SASRec.
- ```--lr```indicates the learning rate of the model for the training.

----

### Hyperparameter configurations

One can refer to ```perpeft_hyperparameter.pickle``` for the optimal hyperparameter configuration of each model on each dataset.
