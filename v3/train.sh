

#!/bin/bash

# 创建一个名为mysession的tmux会话，并在其中执行命令
# tmux new-session -d -s divmode "CUDA_VISIBLE_DEVICES=0,1 torchrun --master-port 29501 --nproc_per_node 2 /data/ruanjh/best_training_method/t5/v3/t5-train-special.py \
# --model_dir /data/ruanjh/best_training_method/t5v1_1-large \
# --data_dir /data/ruanjh/best_training_method/t5 \
# --output_path /data/ruanjh/t5-v4-ce-divmode \
# --ce True \
# --div_mode True --clm True> t5-v4-ce-divmode.log   2>t5-v4-ce-divmode.err"


# tmux new-session -d -s nondivmode01 "CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port 29499 --nproc_per_node 2 /data/ruanjh/best_training_method/t5/v3/t5-train-special.py \
# --model_dir /data/ruanjh/best_training_method/t5v1_1-large \
# --data_dir /data/ruanjh/best_training_method/t5 \
# --output_path /data/ruanjh/t5-v4-ce-005 \
# --ce True \
# --zero_prob 0.05 \
# --div_mode False --clm True > t5-v4-ce-005.log   2>t5-v4-ce-005.err"

#-------------------------------------------------------------------------------------------

tmux new-session -d -s divmode2 "CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port 29601 --nproc_per_node 2 /data/ruanjh/best_training_method/t5/v3/t5-train-special.py \
--model_dir /data/ruanjh/best_training_method/t5v1_1-large \
--data_dir /data/ruanjh/best_training_method/t5 \
--output_path /data/ruanjh/t5-v4-ce-5gram-005-accumulation1-lr2e5 \
--ce True \
--zero_prob 0.05 \
--learning_rate 2e-5 \
--div_mode False --clm True> t5-v4-ce-5gram-005-accumulation1-lr2e5.log   2>t5-v4-ce-5gram-005-accumulation1-lr2e5.err"


tmux new-session -d -s nondivmode02 "CUDA_VISIBLE_DEVICES=0,1 torchrun --master-port 29699 --nproc_per_node 2 /data/ruanjh/best_training_method/t5/v3/t5-train-special.py \
--model_dir /data/ruanjh/best_training_method/t5v1_1-large \
--data_dir /data/ruanjh/best_training_method/t5 \
--output_path /data/ruanjh/t5-v4-ce-5gram-005-accumulation1 \
--ce True \
--zero_prob 0.05 \
--div_mode False --clm True > t5-v4-ce-5gram-005-accumulation1.log   2>t5-v4-ce-5gram-005-accumulation1.err"

# 输出tmux会话已经启动
echo "tmux会话已启动。"

# 可选：显示tmux会话列表，确认会话已创建
tmux ls

# # 可选：等待一段时间后再连接tmux会话，确保命令已经启动
# sleep 3

# # 连接到tmux会话
# tmux attach-session -t mysession

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 /data/ruanjh/best_training_method/t5/v3/t5-train-special.py \
#     --model_dir /data/ruanjh/best_training_method/t5v1_1-large \
#     --data_dir  /data/ruanjh/best_training_method/t5 \
#     --output_path /data/ruanjh/t5-v4-ce-4gram\
#     --ce True \
#     --div_mode False \