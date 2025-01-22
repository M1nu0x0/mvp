#!/usr/bin/env bash

# This script is used to run the inference of the model.
python -m torch.distributed.launch --nproc_per_node=1 --use_env run/inference.py --cfg configs/panoptic/best_model_config.yaml --model_path models/model_best_5view.pth.tar