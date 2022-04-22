#!/bin/bash
GPU=0
LOSS=mse
EXP_NAME=-1
LOG2=True
LR=0.00005

# Models without ComBat or ComBat_seq batch correction 
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME

# Models with ComBat or ComBat_seq batch correction
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq True  --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True  --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq True  --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat True  --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat True  --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat True  --ComBat_seq False --log2 $LOG2  --loss $LOSS --lr $LR --exp_name $EXP_NAME
