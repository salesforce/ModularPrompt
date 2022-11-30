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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
# for baseline adapter
from transformers.activations import get_activation
if torch.__version__ == '1.7.0': # original env, no opendelta installed
    AdapterModel, CompacterModel, PrefixModel = None, None, None
else:
    from opendelta import AdapterModel, CompacterModel, PrefixModel

class T5Adapter(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5Adapter, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        if args.use_lm_adapted == 1:
            print("use lm adapted model!")
            t5ckpt = torch.load(args.lm_adapted_path)
            self.model.load_state_dict(t5ckpt)
            
        if args.debug:
            import pdb;pdb.set_trace()
        if self.args.delta_type == 'adapter':
            self.delta_model = AdapterModel(self.model) # register a delta model to self.model
        elif self.args.delta_type == 'prefix':
            self.delta_model = PrefixModel(self.model, args.prompt_length)
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.freeze_parameters()

    def freeze_parameters(self):
        self.delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True) # only save trained parameters
    

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        return self.model(
            inputs_embeds=input_embed_part,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
    def forward(self, batch):
        lm_labels = batch["target_ids"]
        #print(self.tokenizer.pad_token_id)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        #print(self.model.config.decoder_start_token_id)
        #print(self.model.config.bos_token_id)
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def _generative_step(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=input_embed_part,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
            max_length=self.args.max_gen_length,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
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

