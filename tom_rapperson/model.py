import json
import re

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def get_model_from_huggingface_pretrained(model_name, vocab_size) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    _resize_embeddings(model, vocab_size)
    return model


def get_model_from_pl_checkpoint(file_path) -> GPT2LMHeadModel:
    checkpoint = torch.load(f=file_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = {re.sub(r'^_model\.', '', name): weights for name, weights in state_dict.items()}
    vocab_size = state_dict['transformer.wte.weight'].size()[0]
    gpt_config = json.loads(checkpoint['hyper_parameters']['gpt_config'])
    model = GPT2LMHeadModel(GPT2Config(**gpt_config))
    _resize_embeddings(model=model, vocab_size=vocab_size)
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
