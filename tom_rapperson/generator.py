import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import GPT2LMHeadModel

from tom_rapperson.encoder import SongsEncoder


class SongsGenerator:
    def __init__(self, model: GPT2LMHeadModel, encoder: SongsEncoder):
        self._model = model.eval()
        self._encoder = encoder

    def __call__(self, prefix, artist_name, n_candidates, gen_n_tokens, temperature, top_k, repetition_penalty):
        input_ids = self._get_input_ids(
            prefix=prefix,
            artist_name=artist_name,
            n_candidates=n_candidates,
        )
        gen_token_ids = self._generate(
            input_ids=input_ids,
            gen_n_tokens=gen_n_tokens,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        gen_token_ids = gen_token_ids.cpu().numpy().tolist()
        candidates = [self._encoder.decode(x).rsplit('\n', 1)[0] for x in gen_token_ids]
        return candidates

    def _get_input_ids(self, prefix, artist_name, n_candidates):
        input_ids = self._encoder.encode(prefix, artist_name)
        input_ids = np.array([input_ids for _ in range(n_candidates)], dtype=np.int32)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self._model.device)
        return input_ids

    def _generate(self, input_ids, gen_n_tokens, top_k, temperature, repetition_penalty):
        gen_token_ids = torch.zeros(len(input_ids), gen_n_tokens, dtype=torch.long, device=self._model.device)
        past_key_values = None

        for i_step in tqdm.trange(gen_n_tokens, desc='Generating'):
            model_out = self._model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = model_out.logits[:, -1, :]
            past_token_ids = torch.cat([gen_token_ids, input_ids], dim=1)
            past_token_ids[past_token_ids == self._encoder.new_line_token_id] = 0
            next_token_ids = _sample_next_token_ids(
                next_token_logits=next_token_logits,
                top_k=top_k,
                temperature=temperature,
                token_ids_to_penalize=past_token_ids,
                penalty=repetition_penalty,
            )
            gen_token_ids[:, i_step] = next_token_ids
            input_ids = next_token_ids.unsqueeze(1)
            past_key_values = model_out.past_key_values
        return gen_token_ids


def _sample_next_token_ids(next_token_logits, top_k, temperature, token_ids_to_penalize, penalty):
    _penalize_specific_tokens(
        token_ids_to_penalize=token_ids_to_penalize,
        logits=next_token_logits,
        penalty=penalty,
    )
    next_token_logits.mul_(1 / temperature)
    top_k_logits, top_k_inds = torch.topk(next_token_logits, top_k)
    top_k_probas = F.softmax(top_k_logits, dim=-1)
    next_input_ids = torch.multinomial(top_k_probas.type(torch.float32), num_samples=1)
    next_input_ids = top_k_inds.gather(-1, next_input_ids).squeeze(-1)
    return next_input_ids


def _penalize_specific_tokens(token_ids_to_penalize, logits, penalty):
    assert penalty >= 1.0
    for i in range(logits.size()[0]):
        ids_to_penalize = token_ids_to_penalize[i]
        _penalize_logits_tensor(logits[i], ids_to_penalize, penalty)


def _penalize_logits_tensor(logits, penalty_idx, penalty):
    if penalty == 1.0:
        return

    idx = torch.unique(penalty_idx)
    logits -= logits.max()

    full_exp = torch.exp(logits)

    e = full_exp[idx]
    sum_e = torch.sum(e)
    s = torch.sum(full_exp) - sum_e

    n = torch.log((e * s) / (penalty * s + penalty * sum_e - sum_e))
    logits[idx] = n
