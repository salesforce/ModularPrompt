import os
import pdb
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class T5Prompt(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5Prompt, self).__init__()
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
        self.prompt_length = 0
        self.prompt_embedding = None
        self.mode = args.concat_mode

    def set_prompt_embedding(self,prompt_length,prompt_embedding):
        self.prompt_length = prompt_length
        self.prompt_embedding = nn.parameter.Parameter(prompt_embedding)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        ##### handle prompt, cal input_embed
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.prompt_embedding.repeat(input_embed_part.size(0), 1, 1)

        mask_prompt = torch.full((attention_mask.shape[0],self.prompt_length),1).to(self.args.device)
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
        # return ret

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def _generative_step(self, batch, prefix_fn=None):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        prompt_embed_repeat = self.prompt_embedding.repeat(input_embed_part.size(0), 1, 1)
        if self.mode == 'right_concat':
            allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        elif self.mode == 'left_concat':
            allembedding = torch.cat([prompt_embed_repeat, input_embed_part], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.prompt_length), 1).to(self.args.device)

        if self.mode == 'right_concat':
            all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        elif self.mode == 'left_concat':
            all_attention_mask = torch.cat([mask_prompt, batch["attention_mask"]], 1)

        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
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
