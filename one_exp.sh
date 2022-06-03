#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
NORM=tpm
LOG2=True
COMBAT=False
COMBAT_SEQ=True
FILTER_TYPE=none
# Graph parameters #######################################
STRING=True
ALL_STRING=False
CONF_THR=0.9
CORR_THR=0.7
# Model parameters ########################################
MODEL=baseline
HIDDEN_CHANN=8
DROPOUT=0.5
FINAL_POOL=none
# Training parameters #####################################
EXP_NAME=string_high_conf_batch_9_toy_model_clamp_dropout_0.5_no_node_pool
LOSS=mse
LR=0.00005
EPOCHS=100
BATCH_SIZE=9
ADV_E_TEST=0.0
ADV_E_TRAIN=0.0
N_ITERS_APGD=50



CUDA_VISIBLE_DEVICES=$GPU python main.py --norm $NORM                 --log2 $LOG2               --ComBat $COMBAT              --ComBat_seq $COMBAT_SEQ    \
                                         --filter_type $FILTER_TYPE   --string $STRING           --all_string $ALL_STRING      --conf_thr $CONF_THR        \
                                         --corr_thr $CORR_THR         --model $MODEL             --hidden_chann $HIDDEN_CHANN  --dropout $DROPOUT          \
                                         --final_pool $FINAL_POOL     --exp_name $EXP_NAME       --loss $LOSS                  --lr $LR                    \
                                         --epochs $EPOCHS             --batch_size $BATCH_SIZE   --adv_e_test $ADV_E_TEST      --adv_e_train $ADV_E_TRAIN  \
                                         --n_iters_apgd $N_ITERS_APGD