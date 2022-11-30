#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import pdb
import numpy as np
import time
import random
import time
import logging
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
from transformers.optimization import Adafactor
import sys
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from models.soft_prompt import T5Prompt
from models.finetune import T5Finetune
from models.modular_prompt import T5ModularPrompt
from models.adapter import T5Adapter
from dataset import *
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from sklearn.metrics import classification_report, f1_score
import pickle5 as pickle
from shutil import copyfile
from copy import copy
from utils import *
import re
from data.wild_stages_config import stages_config

tosavepath = "./output"

def set_args():
    parser = argparse.ArgumentParser(description="StPT")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")
    ## 1. dataset
    parser.add_argument("--dataset", dest="dataset", choices=['fewNERD','huffpost', 'fewrel'],
                        default='huffpost', help='choice of dataset, affecting prompt initialization')
    parser.add_argument("--toptokens", dest="toptokens", choices=['c4'],
                        default='c4', help='choice of dataset to extract top tokens for prompt embedding init')
    parser.add_argument("--train_file_name", dest="train_file_name", type=str,
                        default="none", help="train data file path")
    parser.add_argument("--valid_file_name", dest="valid_file_name", type=str,
                        default="none", help="valid data file path")
    parser.add_argument("--test_file_name", dest="test_file_name", type=str,
                        default="none", help="test data file path")                 

    ## 2. prompt trainig
    parser.add_argument("--concat_mode", dest="concat_mode", choices=['left_concat', 'right_concat'],
                        default='right_concat', help='append prompt to the left or right')
    parser.add_argument("--batch_seen_labels", action="store_true",
                        help="only choose label prompts of which the labels are present in current batch")  
    parser.add_argument("--order_inv", action="store_true",
                        help="apply order invariance consistency training")  
    parser.add_argument("--subset_inv", action="store_true",
                        help="apply subset invariance consistency training")
    parser.add_argument("--subset_inv_type", choices=['batch_gt', 'sample_gt', 'length_gt'],
                        default='batch_gt', help="which gt to keep")
    parser.add_argument("--mean_prob", dest="mean_prob", type=float, default=1.0,
                        help="probability to select mean distribution when subset inv type is length_gt")        
    parser.add_argument("--subset_drop_prob", dest="subset_drop_prob", type=float, default=0.25,
                        help="probability to drop a non ground truth label in prompt during training, only effective when args.subset_inv is turned on")
    parser.add_argument("--test_analysis", dest="test_analysis", choices=['none', 'grounding', 'order', 'subset', 'kv_mapping', 'subsetv2', 'drop_random'],
                        default='none', help='if to further analysis inference with diff prompts')
    parser.add_argument("--use_lm_adapted", dest="use_lm_adapted", type=int,
                        default=1, help="whether to use lm_adapted model")
    parser.add_argument("--lm_adapted_path", dest="lm_adapted_path", type=str,
                        default="/export/home/prompting/lm_adapted_models/t5.1.1.lm100k.large/pytorch_model.bin",
                        help="The path of lm_adapted model")
    parser.add_argument("--prompt_length", dest="prompt_length", type=int,
                        default=100, help="The number of prompt")
    parser.add_argument("--prompt_length_label", dest="prompt_length_label", type=int,
                        default=10, help="The number of prompt")
    parser.add_argument("--label_data_initialization", action="store_true",
                        help="if to use label's data to re-init label prompts")

    ## 3. in-the-wild learning
    parser.add_argument("--wild_do_test", choices=['none', 'current', 'all_seen', 'specific'],
                        default='none', help="whether do testing on in-the-wild learning setting (test after each training stage)")
    parser.add_argument("--test_stage_num", type=int,
                        default=-1, help="stage to test")                        
    parser.add_argument("--wild_version", dest="wild_version", choices=['train', 'multitask', 'fused'],
                        default='train', help="in-the-wild stages setting")
    parser.add_argument("--wild_resume_stage_num", dest="wild_resume_stage_num", type=int,
                        default=0, help="number of training stage to resume for in-the-wild learning")
    parser.add_argument("--wild_end_stage_num", dest="wild_end_stage_num", type=int,
                        default=1000, help="number of training stage to end for in-the-wild learning")                      
    parser.add_argument("--enable_forward_transfer", action="store_true",
                        help="whether to support forward transfer in label prompt initialisation")
    parser.add_argument("--forward_transfer_type", choices=['label_embedding_similarity_v2'],
                        default='label_embedding_similarity_v2', help="various ways to support forward transfer")
    parser.add_argument("--forward_transfer_similarity_type", choices=['mean', 'top1', 'top3'],
                        default='mean', help="various ways to use similarity based initilisation")                 
                        
    # adapter baseline
    parser.add_argument("--adapter_encoder_layers", type=int,
                        default=12, help="number of encoder layers of T5")   
    parser.add_argument("--adapter_decoder_layers", type=int,
                        default=12, help="number of decoder layers of T5")
    parser.add_argument("--enable_self_attn_adapter", action="store_true",
                        help="whether to add adapter in self attn layer") 
    parser.add_argument("--input_dim", type=int,
                        default=1024, help="internal feature dim as input for adapter modules")
    parser.add_argument("--reduction_factor", type=int,
                        default=32, help="adapter reduction factor")
    parser.add_argument("--adapters_non_linearity", type=str,
                        default='swish', help="act in adapter")
    parser.add_argument("--weight_init_range", type=float,
                        default=1e-2, help="weight_init_range for adapter")
    parser.add_argument("--delta_type", type=str,
                        default='adapter', help="type of delta model")                    
    
    ## 4. general training
    parser.add_argument("--cache_dir", dest="cache_dir", type=str,
                        default='/export/home/cache', help="The path to store transformers' model cache")
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument("--load_ckpt", dest="load_ckpt", type=int,
                        default=0, help="whether load ckpt before training")
    parser.add_argument("--ckpt_path", dest="ckpt_path", type=str,
                        default='', help="The path to prompt ckpt")
    parser.add_argument("--optimizer", dest="optimizer", choices=['adamW', 'Adafactor', 'SGD'],
                        default='Adafactor', help='choice of optimizer')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                        default=16, help="batch size per gpu")
    parser.add_argument("--valid_size_per_gpu", dest="valid_size_per_gpu", type=int,
                        default=24, help="valid size per gpu")
    parser.add_argument("--test_size_per_gpu", dest="test_size_per_gpu", type=int,
                        default=24, help="test size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=5, help="max epoch number")
    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=0, help="dataloader num_workers")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    parser.add_argument("--model", dest="model", type=str,
                        default="T5Prompt", help="{T5Prompt}")
    parser.add_argument("--model_name", dest="model_name", type=str,
                        default="google/t5-v1_1-large", help="{t5-base,google/t5-v1_1-base}")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=128, help="max sentence length")
    parser.add_argument("--max_gen_length", dest="max_gen_length", type=int,
                        default=16, help="max generation sentence length")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default = 1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1.0, help="max grad norm")
    
    # logging and evaluating / saving
    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=100000, help="step to save")
    parser.add_argument("--log_step", dest="log_step", type=int,
                        default=1, help="how many steps to log")
    parser.add_argument("--eval_step", dest="eval_step", type=int,
                        default=1000000, help="how many steps to eval")
    parser.add_argument("--eval_start_epoch", dest="eval_start_epoch", type=int,
                        default=50, help="after how many epochs to start evaluating")
    parser.add_argument("--eval_epoch", dest="eval_epoch", type=int,
                        default=2, help="how many epochs to eval once")
    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="t5_ckpt", help="ckpt dir to save")           
    parser.add_argument("--log_dir", dest="log_dir", type=str,
                        default='./logs', help="The path to log dir")
    parser.add_argument("--log_name", dest="log_name", type=str,
                        default='dummy', help="The file name of log file")
    parser.add_argument("--save_test", action="store_true",
                        help="if to save test predictions to file")
    parser.add_argument("--save_test_dir", dest="save_test_dir", type=str,
                        default='./saved_test', help="The path to log test preds")
    parser.add_argument("--verbose", action="store_true",
                        help="if true, print out verbose logs")
    parser.add_argument("--test_verbose", action="store_true",
                        help="if true, print out verbose test logs")
    parser.add_argument("--test_only", action="store_true",
                        help="only do test")
    parser.add_argument("--debug", action="store_true",
                        help="enable debugging mode")
    

    args = parser.parse_args()
    args.sequence_labeling_datasets = ['fewNERD']
    args.classification_datasets = ['huffpost', 'fewrel']
    return args

def set_logger(args):
    global logger
    logger = logging.getLogger('root')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = '%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/{args.log_name}.log"),
            logging.StreamHandler()
        ]
    )

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_prompt_from(ckpt_path, model):
    allckpt = torch.load(ckpt_path)
    if args.model == 'T5Prompt':
        model.prompt_length = allckpt["prompt_length"]
        model.prompt_embedding = allckpt["prompt_embedding"]
    elif args.model == 'T5ModularPrompt':
        model.prompt_dict = allckpt['prompt_dict']
        model.prompt_fix_dict = allckpt['prompt_fix_dict']
        for k, v in model.prompt_fix_dict.items():
            model.prompt_fix_dict[k] = v.to(args.device)
    elif args.model == 'T5Finetune':
        model_state_dict = {}
        model = model.cpu()
        for k,v in allckpt['t5-model'].items():
            model_state_dict['model.'+k] = v
        model.load_state_dict(model_state_dict)
        del model_state_dict
        torch.cuda.empty_cache()
    elif args.model == 'T5Adapter':
        model.model.load_state_dict(allckpt['adapters'], strict=False)

def load_prompt(args, model):
    load_prompt_from(args.ckpt_path, model)

def save_model(modeltoeval, optimizer, args, steps):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    if not os.path.exists(tosavepath):
            os.mkdir(tosavepath)
    if not os.path.exists(tosavepath + "/" + args.save_dir):
        os.mkdir(tosavepath + "/" + args.save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    if args.model == 'T5Prompt':
        ckpt = {
            "prompt_length": model_to_save.prompt_length,
            "prompt_embedding": model_to_save.prompt_embedding,
        }
    elif args.model == 'T5ModularPrompt':
        ckpt = {
            "prompt_dict": model_to_save.prompt_dict,
            "prompt_fix_dict": model_to_save.prompt_fix_dict,
        }
    elif args.model == 'T5Finetune':
        ckpt = {
            't5-model': model_to_save.model.state_dict(),
        }
    elif args.model == 'T5Adapter':
        ckpt = {
            "adapters": model_to_save.model.state_dict()
        }
    logger.info("about to save")
    torch.save(ckpt, os.path.join(tosavepath + "/" + args.save_dir, "ckptofT5_"+str(steps)))
    logger.info("ckpt saved")

def dooneeval(modeltoeval,valid_dataloader,args,result_dict,optimizer,i,to_save=True):
    if isinstance(modeltoeval, torch.nn.parallel.DistributedDataParallel):
        model = modeltoeval.module
    else:
        model = modeltoeval
    model.eval()
    allytrue = []
    allypred = []
    correctnum, allnum = 0, 0
    with torch.no_grad():
        logger.info(len(valid_dataloader))
        
        for step, batch in enumerate(valid_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                    "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            
            sen, target, preds = model._generative_step(inputs)
            if args.dataset in args.sequence_labeling_datasets:
                tarres, predres = getonebatchresult_ner(sen,target,preds)
            elif args.dataset in args.classification_datasets:
                tarres, predres = getonebatchresult_classification(sen,target,preds)
            allytrue.extend(tarres)
            allypred.extend(predres)
            thisbatchnum = len(sen)
            for k in range(thisbatchnum):
                allnum += 1
                if target[k].lower() == preds[k].lower():
                    correctnum += 1

    if args.dataset in args.sequence_labeling_datasets:
        f1score = seq_f1_score(allytrue, allypred)
    elif args.dataset in args.classification_datasets:
        f1score = f1_score(allytrue, allypred, average='micro')
    accuracy = float(correctnum) / float(allnum)
    logger.info("allnum: %d", allnum)
    logger.info("correctnum: %d",correctnum)
    logger.info("Accuray: %f",accuracy)
    logger.info('----Validation Results Summary----')
    logger.info(len(allypred))
    logger.info(f1score)
    result_dict['val_F1'].append(f1score)
    if result_dict['val_F1'][-1] > result_dict['best_val_F1'] and to_save:
        logger.info("{} epoch, best epoch was updated! valid_F1: {: >4.5f}".format(i,result_dict['val_F1'][-1]))
        result_dict["best_val_F1"] = result_dict['val_F1'][-1]
        if not os.path.exists(tosavepath):
            os.mkdir(tosavepath)
        if not os.path.exists(tosavepath + "/" + args.save_dir):
            os.mkdir(tosavepath + "/" + args.save_dir)
        
        save_model(model, optimizer, args, 'best')
    return f1score

def get_dataloader(num_workers,dataset, batch_size, max_len, pad_id, sampler):
    collate_fn = SmartBatchingCollate(
        max_length=max_len,
        pad_token_id=pad_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        #shuffle=True, #####?????
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def test(args, test_dataset, seen_labels=[]):
    if args.verbose:
        logger.info(f"test file {test_dataset.filename}")
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = get_dataloader(args.num_workers, test_dataset, args.test_size_per_gpu, args.max_length,
                                      test_dataset.tokenizer.pad_token_id,test_sampler)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    # load ckpt
    if args.model == 'T5Prompt':
        model = T5Prompt(args, t5model, tokenizer)
    elif args.model == 'T5ModularPrompt':
        model = T5ModularPrompt(args, t5model, tokenizer)
    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
    elif args.model == 'T5Adapter':
        model = T5Adapter(args, t5model, tokenizer)
    
    if args.ckpt_path and args.load_ckpt and args.test_only:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(tosavepath, args.save_dir, "ckptofT5_best")
    load_prompt_from(ckpt_path, model)
    
    logger.info("load finished!")

    model.to(args.device)
    model.eval()
    allytrue = []
    allypred = []
    allx = []
    all_target, all_pred = [], []

    
    if args.model == 'T5ModularPrompt':
        if seen_labels:
            model.add_seen_labels(seen_labels)
            if args.verbose:
                logger.info(f'add seen labels {seen_labels}')

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
            
            
            if args.model == 'T5ModularPrompt' and args.subset_inv:
                sen, target, preds = model._generative_step(inputs, labels_set=batch[4])
            else:
                sen, target, preds = model._generative_step(inputs)

            if args.dataset in args.sequence_labeling_datasets:
                tarres, predres = getonebatchresult_ner(sen,target,preds)
            elif args.dataset in args.classification_datasets:
                tarres, predres = getonebatchresult_classification(sen,target,preds)
            allytrue.extend(tarres)
            allypred.extend(predres)
            allx.extend(sen)
            
            all_target.extend(target)
            all_pred.extend(preds)
            
            if step % 10 == 0:
                logger.info("Finished %s steps", step)
            
    if args.dataset in args.sequence_labeling_datasets:
        report = seq_classification_report(allytrue, allypred, digits=4)
    elif args.dataset in args.classification_datasets:
        report = classification_report(allytrue, allypred, digits=4)
    logger.info("\n%s", report)
    if args.test_verbose:
        logger.info(f'targets: {all_target}')
        logger.info(f'predicts: {all_pred}')
    if args.save_test:
        save_test(all_target, all_pred, allx, args)
   
def train(args, model, train_dataset, valid_dataset, prev_fishers=None, prev_params=None, jump_steps=None, prev_valid_datasets=None, wild_stages=None):
    # total step
    step_tot = int(0.5 + len(
        train_dataset) / float(args.gradient_accumulation_steps) / args.batch_size_per_gpu / args.n_gpu) * args.max_epoch
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(
        train_dataset)
    
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = get_dataloader(args.num_workers, train_dataset, args.batch_size_per_gpu, args.max_length,
                                      train_dataset.tokenizer.pad_token_id,train_sampler)
    valid_dataloader = get_dataloader(args.num_workers, valid_dataset, args.valid_size_per_gpu, args.max_length,
                                      valid_dataset.tokenizer.pad_token_id,valid_sampler)

    if prev_valid_datasets is not None:
        prev_valid_dataloaders = []
        for prev_val_dataset in prev_valid_datasets:
            prev_valid_sampler = SequentialSampler(prev_val_dataset)
            prev_val_dataloader = get_dataloader(args.num_workers, prev_val_dataset, args.valid_size_per_gpu, args.max_length,
                                      prev_val_dataset.tokenizer.pad_token_id,prev_valid_sampler)
            prev_valid_dataloaders += [prev_val_dataloader]
    

    base_optimizer_arguments = {"lr": args.lr, "clip_threshold": args.max_grad_norm, "decay_rate": -0.8,
                                "weight_decay": args.weight_decay,
                                "scale_parameter": False, "relative_step": False}
    if args.optimizer == 'adamW':
        optimizer = AdamW
        base_optimizer_arguments = {"lr": args.lr, "weight_decay": args.weight_decay}
    elif args.optimizer == 'Adafactor':
        optimizer = Adafactor
    elif args.optimizer == 'SGD':
        optimizer = SGD 
        base_optimizer_arguments = {"lr": args.lr, "momentum": 0.9}
    
    optimizer = optimizer(params=filter(lambda p: p.requires_grad, model.parameters()), **base_optimizer_arguments)

    model.train()

    startepoch = 0
    Best_F1 = 0.0

    logger.info("Begin train...")
    logger.info("We will train model in %d steps" % step_tot)

    result_dict = {
        'epoch': [],
        'val_F1': [],
        'best_val_F1': Best_F1
    }
    global_step = 0
    model.eval()
    model.train()
    i = startepoch
    enable_random=True
    valid_f1_list = []
    while(i < startepoch + args.max_epoch):
        model.train()
        result_dict['epoch'] = i
        allloss = []

        for step, batch in enumerate(train_dataloader, start=0):
            # jump label embedding training steps (avoid repeated training)
            if jump_steps is not None and global_step <= jump_steps:
                global_step += 1
                continue
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                      "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device), "enable_random":enable_random}
            
            if args.model == 'T5ModularPrompt' and args.subset_inv:
                loss = model(inputs, labels_set=batch[4])
            else:
                loss = model(inputs)
            
            allloss.append(loss.item())
            
            finalloss = loss
            finalloss.backward()
            
            #logger.info(f'{step}: {loss.item()}')
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    #logger.info("step: %d, shcedule: %.3f, loss: %.6f" % (global_step, global_step/step_tot, np.average(allloss)))
                    logger.info("step: %d, schedule: %.3f, loss: %.6f, epoch: %d" % (
                    global_step, global_step / step_tot, np.average(allloss), i))

                if args.local_rank in [0, -1] and global_step % args.eval_step == 0:
                    if prev_valid_datasets is not None:
                        for prev_dataloader in prev_valid_dataloaders:
                            dooneeval(model,prev_dataloader,args,result_dict,optimizer,i, to_save=False)
                    valid_f1 = dooneeval(model,valid_dataloader,args,result_dict,optimizer,i)
                    model.train()

        if args.local_rank in [0, -1] and ((i >= args.eval_start_epoch and i % args.eval_epoch == 0) or (i+1) == startepoch + args.max_epoch):
            valid_f1 = dooneeval(model,valid_dataloader,args,result_dict,optimizer,i)
            valid_f1_list.append(valid_f1)
            if valid_f1 > 0.9999:
                logger.info(f"early stopping at epoch {i}")
                i = startepoch + args.max_epoch - 1
            nums_eval_to_prev_best = valid_f1_list[::-1].index(max(valid_f1_list))
            if nums_eval_to_prev_best >= 5:
                logger.info(f'validation results not improved for 5 evals, stopping now')
                i = startepoch + args.max_epoch - 1
            model.train()
        
        if args.local_rank in [0, -1] and global_step % args.save_step == 0:
            save_model(model, optimizer, args, global_step)
            model.train()
        
        i += 1
    logger.info('finish training')
    if args.local_rank in [0, -1]:
        save_model(model, optimizer, args, global_step)

def train_wild(args, model, train_datasets, valid_datasets, test_datasets, wild_stages):
    
    # grads_means, grad_fishers
    prev_params = []
    prev_fishers = []
    all_seen_labels = set() # {label_name}

    for stage_num, (train_dataset, valid_dataset, test_dataset) in enumerate(zip(train_datasets, valid_datasets, test_datasets)):
        if stage_num < args.wild_resume_stage_num:
            all_seen_labels.update(wild_stages[stage_num])
            continue
        if stage_num > args.wild_end_stage_num:
            continue

        # add current labels for next stage
        if args.model == 'T5ModularPrompt':
            model.reset_seen_labels() # reset previous stage labels
            model.add_seen_labels(wild_stages[stage_num])
            # re-init new label prompts 
            if args.label_data_initialization:
                label_tokens = get_label_tokens(args, train_dataset)
                cur_prompt_length = args.prompt_length_label
                # embedding transfer from previous label prompts
                if args.enable_forward_transfer and stage_num > 0:
                    if args.forward_transfer_type == 'label_embedding_similarity_v2':
                        labels_to_update = wild_stages[stage_num] - all_seen_labels # only init previously unseen labels [in case new task share labels with previous tasks]
                        label_name_embs = get_mix_prompt_l_embedding_v2(args, model, train_dataset.tokenizer, labels_to_update, all_seen_labels)
                else:
                    logger.info(f'random number test: {random.random()}')
                    label_name_embs = get_mix_prompt_l_embedding_v1(args, model, train_dataset.tokenizer, cur_prompt_length, label_tokens, all_seen_labels, wild_stages[stage_num])
                    label_init_stats(args, label_name_embs)
                
                model.set_prompt_embedding(label_name_embs)
        
        all_seen_labels.update(wild_stages[stage_num])

        train(args, model, train_dataset, valid_dataset)
        logger.info(f'finish training stage {str(stage_num)}')
        
        
        if args.local_rank in [0, -1]:
            ckpt_file = os.path.join(tosavepath, args.save_dir, "ckptofT5_best")
            wild_ckpt_file = ckpt_file+f'_{str(stage_num)}'
            copyfile(ckpt_file, wild_ckpt_file)
        if args.wild_do_test != 'none':
            if args.wild_do_test == 'current':
                test(args, test_dataset, wild_stages[stage_num])
            elif args.wild_do_test == 'all_seen':
                for i in range(stage_num+1):
                    logger.info(f"in-the-wild do test on stage {i}")
                    all_labels = [wild_stages[j] for j in range(stage_num+1)]
                    seen_labels_so_far = set().union(*all_labels)
                    if args.verbose:
                        logger.info(seen_labels_so_far)
                    test(args, test_datasets[i], seen_labels_so_far)
                # do task specific inference 
                seen_labels_so_far = wild_stages[stage_num]
                if args.verbose:
                    logger.info(seen_labels_so_far)
                test(args, test_datasets[stage_num], seen_labels_so_far)
        # load from last best trained ckpt
        load_prompt_from(wild_ckpt_file, model)
        model.to(args.device)


    logger.info('finish training')

if __name__ == "__main__":
    args = set_args()
    set_logger(args)
    # print args
    logger.info(args)
    
    # set cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    args.n_gpu = len(args.cuda.split(","))
    #set_seed(args)
    seed_everything(args)

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name,cache_dir=args.cache_dir)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name,cache_dir=args.cache_dir)
    logger.info(len(tokenizer))

    if args.model == "T5Prompt":
        model = T5Prompt(args,t5model,tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            print ("loading from ckpt path", args.ckpt_path)
        else:
            prompt_length = args.prompt_length
            prompt_embedding = get_prompt_embedding(args, model, tokenizer, prompt_length)
            model.set_prompt_embedding(prompt_length, prompt_embedding)
        model.to(args.device)
    
    elif args.model == 'T5ModularPrompt':
        model = T5ModularPrompt(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            model.unfreeze_all_labels()
            model.to(args.device)
            print ("loading from ckpt path", args.ckpt_path)
        else:
            label_name_embs = get_mix_prompt_embedding(args, model, tokenizer, args.prompt_length_label)
            model.to(args.device)
            model.set_prompt_embedding(label_name_embs)
        

    elif args.model == 'T5Finetune':
        model = T5Finetune(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            print ("loading from ckpt path", args.ckpt_path)
        
        model.to(args.device)
    
    elif args.model == 'T5Adapter':
        model = T5Adapter(args, t5model, tokenizer)
        if args.ckpt_path and args.load_ckpt:
            load_prompt(args, model)
            print ("loading from ckpt path", args.ckpt_path)
        model.to(args.device)

    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")

    # continue learning setting
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    wild_stages = stages_config[args.dataset][args.wild_version]

    logger.info(f"wild_stages: {wild_stages}")
    for stage_num,_ in enumerate(wild_stages):
        if not args.test_only:
            train_datasets += [T5Dataset(args.train_file_name.replace('$','_'+str(stage_num)), args.max_length, tokenizer, subset_inv=args.subset_inv)]
            valid_datasets += [T5Dataset(args.valid_file_name.replace('$','_'+str(stage_num)), args.max_length, tokenizer, subset_inv=args.subset_inv)]
        test_datasets += [T5Dataset(args.valid_file_name.replace('$','_'+str(stage_num)), args.max_length, tokenizer, subset_inv=args.subset_inv)]
    test_dataset = T5Dataset(args.test_file_name, args.max_length, tokenizer, subset_inv=args.subset_inv)
    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    
    ## training
    if not args.test_only:
        train_wild(args, model, train_datasets,valid_datasets,test_datasets,wild_stages)
    
    ## testing
    if args.local_rank in [0, -1]:
        if args.test_only and args.wild_do_test != 'none':
            for stage_num in range(len(wild_stages)):
                if args.wild_do_test == 'current': # task specific testing
                    test(args, test_datasets[stage_num], wild_stages[stage_num])
                elif args.wild_do_test == 'all_seen': # task agnostic whole testing
                    if args.wild_resume_stage_num > stage_num:
                        continue # skip to resume stage
                    if args.wild_end_stage_num <= stage_num:
                        break
                    for i in range(stage_num+1):
                        logger.info(f"in-the-wild do test on stage {i}")
                        all_labels = [wild_stages[j] for j in range(stage_num+1)]
                        seen_labels_so_far = set().union(*all_labels)
                        if args.verbose:
                            logger.info(seen_labels_so_far)
                        test(args, test_datasets[i], seen_labels_so_far)
                    # do task specific inference 
                    seen_labels_so_far = wild_stages[stage_num]
                    if args.verbose:
                        logger.info(seen_labels_so_far)
                    test(args, test_datasets[stage_num], seen_labels_so_far)
                elif args.wild_do_test == 'specific' and stage_num == args.test_stage_num: # task specific testing for certain stage
                    test(args, test_datasets[stage_num], wild_stages[stage_num])
        else:
            if args.wild_do_test != 'none' and not args.test_only:
                pass 
            else:
                test(args,test_dataset)
    logger.info("Finish training and testing!")

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

