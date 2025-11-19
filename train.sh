#!/bin/bash

#SBATCH --job-name=thangnvt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=23010572@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate nt_env


python train_cpo.py \
  --pretrained_lora_path ./models/mt5-lora-full/checkpoint-7728 \
  --gradient_checkpointing \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --cpo_beta 0.1 \
  --output_model_dir ./models/mt5-cpo-full