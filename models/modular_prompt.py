#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import pdb
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
import random
from collections import defaultdict
import numpy as np

class T5ModularPrompt(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5ModularPrompt, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        if args.use_lm_adapted == 1:
            print("use lm adapted model!")
            t5ckpt = torch.load(args.lm_adapted_path)
            self.model.load_state_dict(t5ckpt)
            ### if prompt tuning, set requires_grad false
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.prompt_dict = nn.ParameterDict() # {label_name: label_soft_tokens}, specially, self.prompt_dict['__task__']: task_soft_tokens
        self.prompt_fix_dict = {}
        self.mode = args.concat_mode
        self.order_inv = args.order_inv
        self.subset_inv = args.subset_inv
        self.subset_drop_prob = args.subset_drop_prob
        self.seen_labels_wild = set() # seen labels so far, under in-the-wild learning
        self.kl_loss_fn = KLDivLoss(reduction="batchmean") # preds in log-probs and targets as probs
        self.labels_to_key = None

    def construct_label_to_key(self):
        def _no_tag(key):
            return key[3:] if ':' in key else key
        self.labels_to_key = defaultdict(list)
        for k in self.prompt_fix_dict.keys():
            self.labels_to_key[_no_tag(k)] += [k]

    def label_to_label_key(self, labels):
        if self.labels_to_key is None:
            self.construct_label_to_key()
        if type(labels[0]) == list: # ner dataset
            return labels
        else:
            return [self.labels_to_key[l] for l in labels]



    def get_seen_label_prompt(self):
        return {k:v for k, v in self.prompt_dict.items() if k in self.seen_labels_wild}
    
    def add_seen_labels(self, labels):
        self.seen_labels_wild.update(labels)

    def freeze_all_labels(self):
        # for debugging purpose
        for k,v in self.prompt_dict.items():
            self.prompt_dict[k].requires_grad = False
    
    def unfreeze_all_labels(self):
        for k,v in self.prompt_dict.items():
            self.prompt_dict[k].requires_grad = True

    def batch_seen_labels(self, labels_set):
        self.reset_seen_labels()
        self.seen_labels_wild.update(labels_set)

    def reset_seen_labels(self):
        self.seen_labels_wild = set()

    def add_all_seen_labels(self):
        self.seen_labels_wild.update(self.prompt_dict.keys())
                
    def set_prompt_embedding(self, label_name_embs):
        for k,v in label_name_embs.items():
            if k == '__task__':
                self.prompt_dict['__task__'] = nn.parameter.Parameter(v.to(self.model.device))
            else:
                self.prompt_dict[k] = nn.parameter.Parameter(v[1].to(self.model.device))
                self.prompt_fix_dict[k] = v[0].to(self.model.device)

    def _constrcut_prompt_batch(self, batchsize, labels_set=None, is_random=True, force_order_inv=False):
        prompt_embs = []
        lengths = []
        max_length = 0
        for idx in range(batchsize):
            prompt_embs.append(self._construct_prompt(labels_set, is_random, force_order_inv).unsqueeze(0))
            max_length = max(max_length, prompt_embs[-1].size(1))
            lengths.append(prompt_embs[-1].size(1))

        for idx in range(batchsize):
            prompt_embs[idx] = nn.functional.pad(prompt_embs[idx], (0,0,0,max_length-prompt_embs[idx].size(1)), "constant", self.tokenizer.pad_token_id)
        return torch.cat(prompt_embs, 0), torch.as_tensor(lengths)
    
    def _construct_prompt_batch_lgrounding(self, batchsize, labels_set=None, is_random=True, is_reverse=True, kv_random_mapping=False, use_v2=False, use_v3=False):
        prompt_embs = []
        lengths = []
        max_length = 0
        for idx in range(batchsize):
            if use_v2:
                prompt_embs.append(self._construct_prompt_lgrounding_v2(labels_set[idx], is_random, is_reverse, kv_random_mapping).unsqueeze(0))    
            elif use_v3:
                prompt_embs.append(self._construct_prompt_lgrounding_v3(labels_set[idx], is_random).unsqueeze(0))    
            else:
                prompt_embs.append(self._construct_prompt_lgrounding(labels_set[idx], is_random, is_reverse, kv_random_mapping).unsqueeze(0))
            max_length = max(max_length, prompt_embs[-1].size(1))
            lengths.append(prompt_embs[-1].size(1))
        for idx in range(batchsize):
            prompt_embs[idx] = nn.functional.pad(prompt_embs[idx], (0,0,0,max_length-prompt_embs[idx].size(1)), "constant", self.tokenizer.pad_token_id)
        return torch.cat(prompt_embs, 0), torch.as_tensor(lengths)

    # drop one random label, no ground truth
    def _construct_prompt_lgrounding_v3(self, gt_label, is_random=False, is_drop_random=True):
        def raw(label_name): # drop dataset tag if have
            if ':' in label_name:
                return label_name[label_name.find(':')+1:]
            else:
                return label_name

        prompt_emb = []
        labels_to_include = []
        # make sure to add gt label prompt if not doing grounding test
        if type(gt_label) == list:
            labels_to_include.extend(gt_label)
        else:
            labels_to_include.append(gt_label)
        non_gt_seen_labels = sorted([label for label in self.seen_labels_wild if label not in gt_label])
        # randomly sample the number of label prompts
        if is_drop_random:
            num_label_prompts = len(non_gt_seen_labels) - 1
        else:
            num_label_prompts = len(non_gt_seen_labels)
        
        labels_to_include.extend(random.sample(non_gt_seen_labels, num_label_prompts))

        labels_to_include = list(set(labels_to_include))

        items = [k for k,v in self.prompt_dict.items()]
        if (is_random and self.order_inv):
            random.shuffle(items)

        for i, label_name in enumerate(items):
            if label_name not in labels_to_include:
                continue
            prompt_emb.append(self.prompt_fix_dict[label_name])
            prompt_emb.append(self.prompt_dict[label_name])

        return torch.cat(prompt_emb, 0)
    
    def _construct_prompt_lgrounding_v2(self, gt_label, is_random=False, is_reverse=True, kv_random_mapping=False):
        def raw(label_name): # drop dataset tag if have
            if ':' in label_name:
                return label_name[label_name.find(':')+1:]
            else:
                return label_name

        prompt_emb = []
        labels_to_include = []
        # make sure to add gt label prompt if not doing grounding test
        if not is_reverse:
            if type(gt_label) == list:
                labels_to_include.extend(gt_label)
            else:
                labels_to_include.append(gt_label)
        non_gt_seen_labels = sorted([label for label in self.seen_labels_wild if label not in gt_label])
        # randomly sample the number of label prompts
        if self.subset_inv and not is_reverse:
            if random.random() < self.args.mean_prob: # half chance use mean distribution
                num_label_prompts = random.randint(0, len(non_gt_seen_labels)) 
            else: # otherwise full
                num_label_prompts = len(non_gt_seen_labels)
        else:
            num_label_prompts = len(non_gt_seen_labels)
        labels_to_include.extend(random.sample(non_gt_seen_labels, num_label_prompts))

        labels_to_include = list(set(labels_to_include))
        
        items = [k for k,v in self.prompt_dict.items()]
        if (is_random and self.order_inv):
            random.shuffle(items)

        for i, label_name in enumerate(items):
            if label_name not in labels_to_include:
                continue
            prompt_emb.append(self.prompt_fix_dict[label_name])
            prompt_emb.append(self.prompt_dict[label_name])

        return torch.cat(prompt_emb, 0)
    
    def _construct_prompt_lgrounding(self, gt_label, is_random=False, is_reverse=True, kv_random_mapping=False):
        def raw(label_name): # drop dataset tag if have
            if ':' in label_name:
                return label_name[label_name.find(':')+1:]
            else:
                return label_name

        prompt_emb = []
        items = [(k,v) for k,v in self.prompt_dict.items()]
        if kv_random_mapping:
            fixed_values = list(self.prompt_fix_dict.values())
            fixed_keys = list(self.prompt_fix_dict.keys())
            fixed_random_keys = sorted(fixed_keys, key=lambda k: random.random())
            keys_random_mapping = {k:k_ for k,k_ in zip(fixed_keys, fixed_random_keys)}

        # order invariance consistency training
        if (is_random and self.order_inv):
            random.shuffle(items)
        
        for i, (k,v) in enumerate(items):
            if k != '__task__' and (not self.seen_labels_wild or k in self.seen_labels_wild):
                # drop gt label 
                if is_reverse:
                    if k in gt_label:
                        continue # drop gt label
                    else:
                        prompt_emb.append(self.prompt_fix_dict[k])
                        prompt_emb.append(v)
                elif kv_random_mapping: 
                    k_ = keys_random_mapping[k]
                    prompt_emb.append(self.prompt_fix_dict[k_])
                    prompt_emb.append(v)
                elif self.subset_inv:
                    if k not in gt_label and random.random() < self.subset_drop_prob:
                        continue # drop this label
                    else:                    
                        prompt_emb.append(self.prompt_fix_dict[k])
                        prompt_emb.append(v)
                
        return torch.cat(prompt_emb, 0)
    
    def _construct_prompt(self, labels_set=None, is_random=True, force_order_inv=False):
        prompt_emb = []
        if labels_set is not None and type(labels_set[0]) == list:
            labels_set = [l for label in labels_set for l in label]
        items = [(k,v) for k,v in self.prompt_dict.items()]
        # order invariance consistency training
        if (is_random and self.order_inv) or force_order_inv:
            random.shuffle(items)
        if self.args.debug:
            import pdb;pdb.set_trace()
        for i, (k,v) in enumerate(items):
            if k != '__task__' and (not self.seen_labels_wild or k in self.seen_labels_wild):
                if is_random and self.subset_inv and labels_set is not None:
                    # drop a label if it's not gt with subset_drop_prob
                    if k not in labels_set and random.random() < self.subset_drop_prob:
                        continue # drop this label
                    else:                    
                        prompt_emb.append(self.prompt_fix_dict[k])
                        prompt_emb.append(v)
                else:
                    prompt_emb.append(self.prompt_fix_dict[k])
                    prompt_emb.append(v)
                   
        return torch.cat(prompt_emb, 0)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None, labels_set=None, enable_random=True, force_order_inv=False
    ):
        if self.args.debug:
            import pdb;pdb.set_trace()
        ##### handle prompt, cal input_embed
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        if enable_random and self.subset_inv:
            if self.args.subset_inv_type == 'batch_gt':
                prompt_embed_repeat, lengths = self._constrcut_prompt_batch(batchsize=input_embed_part.size(0), labels_set=labels_set, force_order_inv=force_order_inv)
                prompt_length = prompt_embed_repeat.size(1)
                mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
            elif self.args.subset_inv_type == 'sample_gt':
                prompt_embed_repeat, lengths = self._construct_prompt_batch_lgrounding(batchsize=input_embed_part.size(0), labels_set=labels_set, is_random=True, is_reverse=False, kv_random_mapping=False)
                prompt_length = prompt_embed_repeat.size(1)
                mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
            elif self.args.subset_inv_type == 'length_gt':
                prompt_embed_repeat, lengths = self._construct_prompt_batch_lgrounding(batchsize=input_embed_part.size(0), labels_set=labels_set, is_random=True, is_reverse=False, use_v2=True)
                prompt_length = prompt_embed_repeat.size(1)
                mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
        #if True:
        else:
            prompt_embedding = self._construct_prompt(is_random=enable_random, labels_set=labels_set, force_order_inv=force_order_inv)
            prompt_embed_repeat = prompt_embedding.repeat(input_embed_part.size(0), 1, 1)
            prompt_length = prompt_embedding.size(0)
            mask_prompt = torch.full((attention_mask.shape[0], prompt_length),1).to(self.args.device)
        if self.mode == 'right_concat':
            allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
            all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        if self.mode == 'left_concat':
            allembedding = torch.cat([prompt_embed_repeat, input_embed_part], 1)
            all_attention_mask = torch.cat([mask_prompt, attention_mask], 1)
        
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )


    def forward(self, batch, labels_set=None):
        if labels_set:
            labels_set = self.label_to_label_key(labels_set)
        if self.args.batch_seen_labels and labels_set is not None:
            old_seen_labels = self.seen_labels_wild.copy()
            self.batch_seen_labels(labels_set)
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            labels_set=labels_set,
            enable_random=batch['enable_random'],
        )
        if self.args.batch_seen_labels and labels_set is not None:
            self.reset_seen_labels()
            self.add_seen_labels(old_seen_labels)

        loss = outputs[0]

        return loss

    
    def _generative_step(self, batch, labels_set=None, prefix_fn=None):
        if labels_set:
            labels_set = self.label_to_label_key(labels_set)
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        if self.args.test_analysis == 'grounding':
            prompt_embed_repeat, lengths = self._construct_prompt_batch_lgrounding(batchsize=input_embed_part.size(0), labels_set=labels_set, is_random=True, is_reverse=True)
            prompt_length = prompt_embed_repeat.size(1)
            mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
        elif self.args.test_analysis == 'drop_random':
            prompt_embed_repeat, lengths = self._construct_prompt_batch_lgrounding(batchsize=input_embed_part.size(0), labels_set=labels_set, use_v3=True)
            prompt_length = prompt_embed_repeat.size(1)
            mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
        elif self.args.test_analysis == 'subsetv2':
            prompt_embed_repeat, lengths = self._construct_prompt_batch_lgrounding(batchsize=input_embed_part.size(0), labels_set=labels_set, is_random=True, is_reverse=False, kv_random_mapping=False)
            prompt_length = prompt_embed_repeat.size(1)
            mask_prompt = (torch.arange(prompt_length).expand(len(lengths), prompt_length) < lengths.unsqueeze(1)).long().to(self.args.device)
        elif self.args.test_analysis == 'order':
            prompt_embedding = self._construct_prompt(is_random=True)
            prompt_embed_repeat = prompt_embedding.repeat(input_embed_part.size(0), 1, 1)
            prompt_length = prompt_embedding.size(0)
            mask_prompt = torch.full((batch["attention_mask"].shape[0], prompt_length), 1).to(self.args.device)
        elif self.args.test_analysis == 'none':
            prompt_embedding = self._construct_prompt(is_random=False)
            prompt_embed_repeat = prompt_embedding.repeat(input_embed_part.size(0), 1, 1)
            prompt_length = prompt_embedding.size(0)
            mask_prompt = torch.full((batch["attention_mask"].shape[0], prompt_length), 1).to(self.args.device)
        
        if self.args.debug:
            import pdb;pdb.set_trace()

        if self.mode == 'right_concat':
            allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
            all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        elif self.mode == 'left_concat':
            allembedding = torch.cat([prompt_embed_repeat, input_embed_part], 1)
            all_attention_mask = torch.cat([mask_prompt, batch["attention_mask"]], 1)
        
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=self.args.max_gen_length,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_fn
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input,target,preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))


    