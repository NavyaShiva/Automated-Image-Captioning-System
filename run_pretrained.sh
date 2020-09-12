#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model Pretrained \
    --epochs 2 \
    --weight-decay 0.0 \
    --momentum 0.99 \
    --batch-size 5 \
    --optimizer adam \
    --hidden-dim 512 \
    --embed-dim 256 \
    --lr 0.01 | tee pretrained.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
