#!/bin/bash
GPU=0
LOSS=mse
EXP_NAME=deepergcn_test_batch_20
LOG2=True
LR=0.00005
EPOCHS=100
CORR_THR=0.7
FILTER_TYPE=none

# Raw models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
