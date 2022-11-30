#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
'''Convert tensorflow lm adapted t5 model to pytorch (huggingface) 
'''
from transformers import T5ForConditionalGeneration
import transformers 
import torch
from torch import nn
import os
import shutil
import argparse

def convert(model_type, ckpt_path, save_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_type)
    lm_adapted_model = transformers.models.t5.modeling_t5.load_tf_weights_in_t5(model, None, ckpt_path)
    lm_adapted_model.save_pretrained(save_dir)
    print(f"Saved {model_type} to {save_dir}")

def convert_model_main(args):
    size = args.size # ['small', 'base', 'large']
    data_dir = args.data_root
    model_type = f'google/t5-v1_1-{size}' #['google/t5-v1_1-small','google/t5-v1_1-base','google/t5-v1_1-large']
    save_dir = f'{data_dir}/t5.1.1.lm100k.{size}/'
    ckpt_path = save_dir+'model.ckpt-1100000'
    convert(model_type, ckpt_path, save_dir)

def set_args():
    parser = argparse.ArgumentParser(description="StPT")
    parser.add_argument("--size", choices=['base', 'small', 'large'], default="base")
    parser.add_argument("--data_root", type=str, default="/export/home/prompting/lm_adapted_models")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_args()
    convert_model_main(args)