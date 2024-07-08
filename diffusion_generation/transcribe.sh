#!/bin/bash

#SBATCH -t 6-23:59:00
#SBATCH --nodes 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --array=0-8

RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=$SLURM_ARRAY_TASK_COUNT \
python fine_tune.py --dataset=imagenet --output_dir=/home/sreyang/scratch.ramanid-prj/da-fusion \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--resolution=512 --train_batch_size=4 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 --max_train_steps=1000 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--only_save_embeds --num-trials 8 --examples-per-class 1 2 4 8 16 