import json
import random
import os
import sys

sys.path.append("..")
import pdb
import re
import pdb
import math
import torch
import numpy as np
import linecache
from pathlib import Path


from collections import Counter
from torch.utils import data
import random
import numpy as np
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader

class T5Dataset(Dataset):
    def __init__(self, filename, maxlen, tokenizer, subset_inv=False, dataset_tag=''):
        super(T5Dataset, self).__init__()
        self.filename = filename
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.data = []
        self.data = self.getalldata(self.filename)
        self.num_entries = len(self.data)
        self.subset_inv = subset_inv
        self.dataset_tag = dataset_tag

    def getalldata(self,filename):
        f = open(filename,'r')
        alldata = []
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            linelist = oneline.split("\t")
            if len(linelist) != 2:
                print(oneline)
                print(linelist)
            onedata = []
            if r"\n" in linelist[0] or r"\t" in linelist[0]:
                linelist[0] = linelist[0].replace(r"\n", "\n").replace(r"\t", '\t')
            onedata.append(linelist[0])
            onedata.append(linelist[1])
            alldata.append(onedata)
            
        f.close()
        return alldata

    def __getitem__(self, idx):
        inputdata = self.data[idx][0]
        targetdata = self.data[idx][1]
        # add subset variation
        if self.subset_inv:
            if ';' in targetdata and '!' in targetdata: # ner format
                gt_labels = list(set([t.split(' ! ')[1] for t in targetdata.rstrip().split(' ;') if t]))
            else:
                gt_labels = targetdata.rstrip('')
                if self.dataset_tag:
                    gt_labels = self.dataset_tag+gt_labels
        else:
            gt_labels = set()

        inputres = self.tokenizer.batch_encode_plus([inputdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        targetres = self.tokenizer.batch_encode_plus([targetdata], padding=False, max_length=self.maxlen, truncation=True, return_tensors="pt")
        
        return inputres["input_ids"].squeeze(), targetres["input_ids"].squeeze(), gt_labels

    def __len__(self):
        return self.num_entries


class SmartBatchingCollate:
    def __init__(self, max_length, pad_token_id):
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):

        sequences, targets, labels = list(zip(*batch))

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        target_ids, target_mask = self.pad_target(targets, max_sequence_length=self._max_length, pad_token_id=self._pad_token_id)

        output = input_ids, attention_mask, target_ids, target_mask, labels
        #output = input_ids, attention_mask, target_ids, target_mask
        return output

    def pad_target(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)    ####whether because max_length is not 512?
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences,attention_masks


    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        ##tokenize sequence_batch
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences = []
        attention_masks = []
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks

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
