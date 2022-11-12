#!/bin/bash
GPU=3
LOSS=mse
EXP_NAME=-1
LOG2=True
LR=0.00005
EPOCHS=100
CORR_THR=0.1
FILTER_TYPE=100var

# Raw models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TPM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TMM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

#####################################################################################################################################################################################################################
FILTER_TYPE=100diff

# Raw models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TPM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TMM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

#####################################################################################################################################################################################################################
FILTER_TYPE=1000var

# Raw models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TPM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TMM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

#####################################################################################################################################################################################################################
FILTER_TYPE=1000diff

# Raw models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm raw --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TPM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tpm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME

# TMM models
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat True  --ComBat_seq False --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
CUDA_VISIBLE_DEVICES=$GPU python main.py --norm tmm --ComBat False --ComBat_seq True  --filter_type $FILTER_TYPE --corr_thr $CORR_THR --log2 $LOG2  --loss $LOSS --lr $LR --epochs $EPOCHS --exp_name $EXP_NAME
