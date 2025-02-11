#!/usr/bin/env bash

torchrun --nproc_per_node=1 --standalone ./run/validate_3d.py --cfg configs/panoptic/best_model_config.yaml --model_path models/model_best_5view.pth.tar
# python -m torch.distributed.launch --nproc_per_node=1 --use_env run/validate_3d.py --cfg configs/panoptic/best_model_config.yaml --model_path models/model_best_5view.pth.tar
