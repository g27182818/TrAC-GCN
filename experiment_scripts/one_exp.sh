#!/bin/bash
# GPU to use
GPU=0
# Dataset parameters #####################################
VAL_FRAC=0.2
TEST_FRAC=0.2
NORM=tpm
LOG2=True
BATCH_CORR=combat
FILTER_TYPE=None
BATCH_SAMPLE_THR=100
EXP_FRAC_THR=0.5
BATCH_NORM=True
SHUFFLE_SEED=0
FORCE_COMPUTE=False
# Graph parameters #######################################
STRING=True
ALL_STRING=False
CONF_THR=0.9
CORR_THR=0.81
P_THR=0.05
# Model parameters ########################################
MODEL=trac_gcn
HIDDEN_CHANN=8
DROPOUT=0.5
FINAL_POOL=None
# Training parameters #####################################
EXP_NAME=misc_test
LOSS=mse
LR=0.00005
EPOCHS=100
BATCH_SIZE=10
ADV_E_TEST=0.0
ADV_E_TRAIN=0.0
N_ITERS_APGD=50


CUDA_VISIBLE_DEVICES=$GPU python main.py    --val_frac $VAL_FRAC            --test_frac $TEST_FRAC          --norm $NORM                            --log2 $LOG2                    \
                                            --batch_corr $BATCH_CORR        --filter_type $FILTER_TYPE      --batch_sample_thr $BATCH_SAMPLE_THR    --exp_frac_thr $EXP_FRAC_THR    \
                                            --batch_norm $BATCH_NORM        --shuffle_seed $SHUFFLE_SEED    --force_compute $FORCE_COMPUTE          --string $STRING                \
                                            --all_string $ALL_STRING        --conf_thr $CONF_THR            --corr_thr $CORR_THR                    --p_thr $P_THR                  \
                                            --model $MODEL                  --hidden_chann $HIDDEN_CHANN    --dropout $DROPOUT                      --final_pool $FINAL_POOL        \
                                            --exp_name $EXP_NAME            --loss $LOSS                    --lr $LR                                --epochs $EPOCHS                \
                                            --batch_size $BATCH_SIZE        --adv_e_test $ADV_E_TEST        --adv_e_train $ADV_E_TRAIN              --n_iters_apgd $N_ITERS_APGD
