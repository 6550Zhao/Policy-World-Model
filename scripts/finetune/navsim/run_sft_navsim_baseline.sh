#!/bin/bash

source activate pwm
export PYTHONPATH=/path/to/root/:$PYTHONPATH
accelerate launch --config_file /path/to/accelerate_configs/2_gpus_deepspeed_zero2.yaml --main_process_port=8924 /path/to/training/fine-tune_navsim.py
