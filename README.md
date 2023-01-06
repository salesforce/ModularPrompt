## Learning Label Modular Prompts for Text Classification in the Wild <a name="corl"></a>


This is the official code for the paper [**Learning Label Modular Prompts for Text Classification in the Wild**](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.109/) (accepted to EMNLP 2022).

Authors:
[Hailin Chen](https://www.linkedin.com/in/chenhailin/), [Amrita Saha](https://scholar.google.co.uk/citations?user=3Zb5Y2YAAAAJ&hl=en), [Shafiq Joty](https://raihanjoty.github.io/), [Steven C.H. Hoi](https://scholar.google.com/citations?user=JoLjflYAAAAJ&hl=en) 

## Installation 
```sh
conda create --name moduPT python=3.6.13
source activate moduPT
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install transformers datasets scikit-learn seqeval pickle5 sentencepiece
```
We use pretrain model T5-large (LM adapted): download [t5.1.1.lm100k.large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.large) from [T5 repo](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md). Put it under `${project_dir}/lm_adapted_models/t5.1.1.lm100k.large`. Then convert it to pytorch checkpoint by
```sh
python convert.py --size large --data_root ${project_dir}/lm_adapted_models
ls ${project_dir}/lm_adapted_models/t5.1.1.lm100k.large/pytorch_model.bin
```


## Datasets 
We experiment with three datasets: [HuffPost News](https://www.kaggle.com/datasets/rmisra/news-category-dataset), [FewNERD](https://ningding97.github.io/fewnerd/) and [FewRel](https://www.zhuhao.me/fewrel/). We randomly sampled fewshots data and split them into multi-stages. You can download the processed data from [google drive](https://drive.google.com/file/d/1n7ihI4EZnToaQhnjSPC64H79L_GK4wlB/view?usp=sharing). Unzip it and put `fewNERD` `fewrel` `huffpost` folders under `$project_root/data` directory.

## Training
Commands for running training & testing of [`modularPrompt` | `PromptTuning` | `Finetuning`] on HuffPost News:
```sh
zsh trainner_hp.sh
zsh trainner_hp_PT.sh
zsh trainner_hp_Finetune.sh
```
Replace `trainner_hp.sh` with `trainner_ner.sh` or `trainner_re.sh` for other datasets.
You might want to change the following arguments in the above scripts:
| **Parameters** |                                                                                **Description**                                                                               
|:--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `--lm_adapted_path`      | path to lm-adapted T5 checkpoint, e.g. `$project_root/lm_adapted_models/t5.1.1.lm100k.large/pytorch_model.bin`
| `--cache_dir`      | directory to store huggingface model cache
| `--mean_prob`      | "probability p" in paper section 3.3: the chance to subsample S from $\Omega$
| `run_idx`      | run id for multi-seed experiments (5 seeds). Default `0`, choose from [0,1,2,3,4]

## License 

The code is released under BSD 3-Clause - see `LICENSE.txt` for details.

This code is developed from other open source projects: [transformers](https://github.com/huggingface/transformers). We thank the original contributors of these works for open-sourcing their valuable source codes. 
