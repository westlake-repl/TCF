# Text-based Collaborative Filtering
<a href="https://arxiv.org/pdf/2305.11700.pdf" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2305.11700-FAA41F.svg?style=flat" /></a>
<!-- <a href="https://dl.acm.org/doi/abs/10.1145/3539618.3591932" alt="CIKM"><img src="https://img.shields.io/badge/CIKM-2025-%23002FA7.svg?style=flat" /></a>  -->
![Text-based Recsys](https://img.shields.io/badge/Task-Text_based_Recsys-red) 
![Foundation Model](https://img.shields.io/badge/Task-Foundation_Model-red) 
![Recoomendation](https://img.shields.io/badge/Task-Recommendation-red) 

Quick links: 
[🗃️Dataset](#Dataset) |
[📭Citation](#Citation) |
[🛠️Reproduce](#Training) |


This repository contains the source code for the **CIKM 2025** paper **''Exploring the upper limits of text-based collaborative filtering using large language models: Discoveries and insights''**.

![](fig/TCF.png) 


Full version in [[Arxiv]](https://arxiv.org/pdf/2305.11700.pdf).

## Dataset

### Data Download 

We have preprocessed the news recommendation dataset (MIND) and text-only dataset (HM, Bili) for the ease of use.

You can download preprocessed datasets from [here](https://drive.google.com/file/d/1L_eUk1ePZY-wylcK6nI0ljZdyImYDFkB/view). Unzip the downloaded file, and put the unzipped directory `data` into the root directory of this repository.

The datasets are organized as follows:

```
data/
├── bl_50k/ # Random selected 50,000 users' interactions from Bili dataset
├── bl_50k_20/ # Selected Bili dataset with items < 20 interactions removed
├── bl_50k_50/ # Selected Bili dataset with items < 50 interactions removed
├── bl_50k_200/ # Selected Bili dataset with items < 200 interactions removed
├── hm_200k/ # Random selected 200,000 users' interactions from HM dataset
├── hm_200k_20/ # Selected HM dataset with items < 20 interactions removed
├── hm_200k_50/ # Selected HM dataset with items < 50 interactions removed
├── hm_200k_200/ # Selected HM dataset with items < 200 interactions removed
├── mind_200k/ # Random selected 200,000 users' interactions from MIND dataset
├── mind_200k_20/ # Selected MIND dataset with items < 20 interactions removed
├── mind_200k_50/ # Selected MIND dataset with items < 50 interactions removed
├── mind_200k_200/ # Selected MIND dataset with items < 200 interactions removed
```

## Training

### Requirements
```
# Create a conda environment
conda create -n tcf python=3.8
conda activate tcf

# Install requirements
pip install -r requirements.txt
 
# Install PyTorch 1.12.1+cu113
wget https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl
rm torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl
```

### Pre-inference item embeddings 

For training efficiency and to stay within CUDA memory for models up to 175B, we precompute item embeddings with chunked, block‑wise inference. This stores intermediate block‑level latent embeddings, trading more time and disk space for much lower peak memory than loading the full model with transformers' `device_map=auto`.

The chunked, block‑wise inference currently only supports the following models:
- opt series
- flan-t5 series
- bert/roberta series

Example: `facebook/opt-125m` on `bl_50k`. To use another model or dataset, adjust the arguments in `pre_inference/run.py`.
To run the pre‑inference script:

```
cd pre_inference
python run.py \
    --accelerator gpu \
    --min_item_seq_len 5 \
    --max_item_seq_len "None" \
    --sasrec_seq_len 20 \
    --tokenized_len 20 \
    --dataset bl_50k \ 
    --plm_name facebook/opt-125m \ 
    --plm_last_n_unfreeze 0 \
    --pre_inference_batch_size 1 \
    --pre_inference_devices 0 \
    --pre_inference_precision 32 \
    --pre_inference_num_workers 4 \
    --pre_inference_layer_wise False
```

About `--plm_last_n_unfreeze`:

- Set it to N to unfreeze the last N layers during training; inference saves latents right before those N layers. If 0, all layers are frozen and embeddings are the final‑layer outputs.

- For very large models that cannot be loaded with `device_map`, choose N based on the model's layer count (see the model card) and your GPU memory. Example: `facebook/opt-175b` has 96 layers. Start with `--plm_last_n_unfreeze 90`: the first run computes up to the embedding layer plus the first 6 blocks and stores the intermediate latents. Then rerun with `80`; the script loads the saved latents and processes the next 10 blocks. Iterate by decreasing N (80, 70, 60, 50, 40, 30, 20, 2, 0) to complete inference.

The inferenced item embeddings will be saved as `data/{dataset}/{dataset}_maxlen@{max_item_seq_len}_minlen@{min_item_seq_len}_toklen@{tokenized_len}_saslen@{sasrec_seq_len}_processed/{model_name}_freeze@{n_total_layers-plm_last_n_unfreeze}_inferenced_embs.pt`

### Training

#### For id-based CF training, you can use the following command

Take `bl_50k` as an example:

```
cd train
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1251 run.py \
    --root_data_dir "../data/" \
    --dataset "bl_50k" \
    --behaviors "behaviors.tsv" \
    --news "videos.tsv" \
    --mode "train" \
    --item_tower "id" \
    --loss_type "IBCE" \
    --split_method "leave_one_out" \
    --epoch 50 \
    --l2_weight 0.1 \
    --drop_rate 0.1 \
    --batch_size 64 \
    --lr 1e-3 \
    --embedding_dim 64 \
    --label_screen "YOUR_EXPERIMENT_NAME" \
    --power 1 \
    --architecture "DSSM" \
    --early_stop_patience 10 \
    --n_project_out_mlp_layers 1 \
    --n_mlp_layers 0
```

#### For text-based CF training, you can use the following command

Take SASRec for `bl_50k` and `facebook/opt-125m` with unfreeze last 2 layers as an example:

```
cd train
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1251 run.py \
    --root_data_dir "../data/" \
    --dataset "bl_50k" \
    --behaviors "behaviors.tsv" \
    --news "videos.tsv" \
    --mode "train" \
    --language_model_name "facebook/opt-125m" \
    --inference_path "../data/bl_50k/bl_50k_maxlen@INF_minlen@5_toklen@30_saslen@20_processed/OPT125M_freeze@10_inferenced_embs.pt" \ # change to the path of your pre-inferenced item embeddings
    --unfreeze_last_n_layer 2 \ # Should be exact same with pre-inference embeddings' unfreeze layer number
    --keep_embedding_layer "True" \ # Whether to keep the embedding layer of the PLM
    --item_tower "modal" \
    --loss_type "IBCE" \
    --split_method "leave_one_out" \
    --epoch 200 \
    --l2_weight 0.1 \
    --fine_tune_l2_weight 0.01 \
    --drop_rate 0.1 \
    --batch_size 64 \
    --lr 1e-3 \
    --fine_tune_lr 5e-5 \
    --embedding_dim 512 \
    --label_screen "YOUR_EXPERIMENT_NAME" \
    --power 1 \
    --architecture "SASRec" \ # DSSM or SASRec
    --transformer_block 2 \ # SASRec's transformer block number
    --num_attention_heads 2 \ # SASRec's attention heads number
    --early_stop_patience 5 \
    --n_project_out_mlp_layers 2 \
    --n_mlp_layers 0 \
```

## Citation
If you use our code or find TCF useful in your work, please cite our paper as:

```bib
@article{li2023exploring,
  title={Exploring the upper limits of text-based collaborative filtering using large language models: Discoveries and insights},
  author={Li, Ruyu and Deng, Wenhao and Cheng, Yu and Yuan, Zheng and Zhang, Jiaqi and Yuan, Fajie},
  journal={arXiv preprint arXiv:2305.11700},
  year={2023}
}
```