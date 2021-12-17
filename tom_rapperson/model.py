import json
from tom_rapperson.unlikelihood_loss import get_unlikelihood_loss
import torch.nn as nn
import re

import torch
from transformers import GPT2Config, GPT2LMHeadModel


class Model(nn.Module):
    def __init__(self, gpt2_model: GPT2LMHeadModel):
        super().__init__()
        self._gpt2_model = gpt2_model
        self._distractor_clf = nn.Linear(self._gpt2_model.config.hidden_size, 2)
        self._calc_distractor_loss = nn.CrossEntropyLoss()

    @property
    def backbone(self):
        return self._gpt2_model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
            self,
            input_ids,
            is_distractor,
            lm_labels,
            cls_token_positions,
            ul_alpha,
    ):
        gpt2_model_output = self.calc_gpt2_model_output(
            input_ids=input_ids,
            lm_labels=lm_labels,
        )
        distractor_logits = self.calc_distractor_logits(
            last_hidden_state=gpt2_model_output.hidden_states[-1],
            cls_token_positions=cls_token_positions,
        )
        distractor_loss = self._calc_distractor_loss(distractor_logits, is_distractor)
        ul_loss = get_unlikelihood_loss(gpt2_model_output.logits, input_ids) * ul_alpha
        lm_loss = gpt2_model_output.loss
        loss = lm_loss + ul_loss + distractor_loss
        return loss, lm_loss, ul_loss, distractor_loss

    def calc_gpt2_model_output(
            self,
            input_ids,
            lm_labels=None,
            past_key_values=None,
            position_ids=None,
    ):
        return self._gpt2_model(
            input_ids=input_ids,
            labels=lm_labels,
            attention_mask=input_ids != 0,
            return_dict=True,
            past_key_values=past_key_values,
            use_cache=True,
            position_ids=position_ids,
        )

    def calc_distractor_logits(self, last_hidden_state, cls_token_positions):
        new_shape = (last_hidden_state.size()[0] * last_hidden_state.size()[1], -1)
        shift = torch.arange(0, new_shape[0], last_hidden_state.size()[1], device=last_hidden_state.device)
        last_hidden_state = last_hidden_state.reshape(new_shape)
        cls_state = last_hidden_state[cls_token_positions + shift]
        distractor_logits = self._distractor_clf(cls_state)
        return distractor_logits

    def calc_distractor_logits_1_token(self, last_hidden_state):
        distractor_logits = self._distractor_clf(last_hidden_state.squeeze(1))
        return distractor_logits


def get_model_from_huggingface_pretrained(model_name, vocab_size) -> Model:
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    _resize_embeddings(model, vocab_size)
    return Model(model)


def get_model_from_pl_checkpoint(file_path) -> Model:
    checkpoint = torch.load(f=file_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = {re.sub(r'^_model\.', '', name): weights for name, weights in state_dict.items()}
    vocab_size = state_dict['_gpt2_model.transformer.wte.weight'].size()[0]
    gpt_config = json.loads(checkpoint['hyper_parameters']['gpt_config'])
    gpt_config['use_past'] = True
    model = GPT2LMHeadModel(GPT2Config(**gpt_config))
    _resize_embeddings(model=model, vocab_size=vocab_size)
    model = Model(model)
    model.load_state_dict(state_dict)
    return model


def _resize_embeddings(model: GPT2LMHeadModel, vocab_size):
    old_size = model.base_model.wte.num_embeddings
    n_new = vocab_size - old_size
    if n_new < 0:
        raise ValueError(f"Can't resize embeddings: new vocab size ({vocab_size}) can not be less than the "
                         f"old embeddings number ({old_size}).")
    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    reference_emb = model.base_model.wte.weight.data.mean(0)
    model.base_model.wte.weight.data[idx:] = reference_emb.unsqueeze(0)
