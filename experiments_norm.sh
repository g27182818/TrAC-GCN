#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python main.py --norm raw --log2 False --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm raw --log2 False --loss l1  --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm raw --log2 True  --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm raw --log2 True  --loss l1  --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tmm --log2 False --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tmm --log2 False --loss l1  --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tmm --log2 True  --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tmm --log2 True  --loss l1  --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tpm --log2 False --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tpm --log2 False --loss l1  --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tpm --log2 True  --loss mse --exp_name -1
CUDA_VISIBLE_DEVICES=3 python main.py --norm tpm --log2 True  --loss l1  --exp_name -1
