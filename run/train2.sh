#!/usr/bin/env bash

# This script is used to run the inference of the model.
torchrun --nproc_per_node=1 --standalone ./run/train_3d.py --cfg configs/panoptic/best_model_config.yaml