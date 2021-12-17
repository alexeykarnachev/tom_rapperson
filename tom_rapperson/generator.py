from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from tom_rapperson.encoder import SongsEncoder
from tom_rapperson.model import Model


class SongsGenerator:
    def __init__(self, model: Model, encoder: SongsEncoder):
        self._model = model.eval().half()
        self._encoder = encoder
        self._eos_token_id = self._encoder.new_line_token_id

    @torch.no_grad()
    def __call__(
            self,
            prefixes: Sequence[str],
            contexts: Sequence[Sequence[str]],
            temperature,
            top_k,
            repetition_penalty,
    ):
        input_ids = self._get_input_ids(prefixes=prefixes, contexts=contexts)
        gen_token_ids, sample_lengths, distractor_scores = self._generate(
            input_ids=input_ids,
            gen_n_tokens=self._encoder.max_n_post_tokens,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        gen_token_ids = gen_token_ids.cpu().numpy().tolist()
        samples_lengths = sample_lengths.cpu().numpy().tolist()
        distractor_scores = distractor_scores.cpu().numpy().tolist()
        candidates = []
        for i in range(len(gen_token_ids)):
            candidate_token_ids = gen_token_ids[i][:samples_lengths[i]]
            candidate = self._encoder.decode(candidate_token_ids)
            distractor_score = distractor_scores[i]
            candidates.append((distractor_score, candidate))
        return candidates

    def _get_input_ids(self, prefixes, contexts):
        assert len(prefixes) == len(contexts)
        input_ids = []
        for prefix, context in zip(prefixes, contexts):
            input_ids.append(self._encoder.encode_inference(prefix, context))
        input_ids = _pad_left(input_ids, pad_value=0).astype(np.int64)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self._model.device)
        return input_ids

    def _generate(self, input_ids, gen_n_tokens, top_k, temperature, repetition_penalty):
        gen_token_ids = torch.zeros(len(input_ids), gen_n_tokens, dtype=torch.long, device=self._model.device)
        past_key_values = None
        sample_lengths = torch.zeros(len(input_ids), dtype=torch.int32, device=self._model.device)
        distractor_scores = torch.zeros_like(sample_lengths, dtype=torch.float16) + 9999.0
        position_ids = torch.arange(input_ids.size()[1]).unsqueeze(0).expand(input_ids.size()[0], -1)
        position_ids = position_ids.to(self._model.device)
        position_ids = position_ids - (input_ids == 0).sum(-1).unsqueeze(-1)
        position_ids = position_ids.clamp(0)
        for i_step in tqdm.trange(gen_n_tokens, desc='Generating'):
            model_out = self._model.calc_gpt2_model_output(
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
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
            ready_for_cls_sample_inds = torch.where(next_token_ids == self._encoder.end_of_target_token_id)[0]
            if len(ready_for_cls_sample_inds):
                last_hidden_state = model_out.hidden_states[-1][ready_for_cls_sample_inds]
                logits = self._model.calc_distractor_logits_1_token(last_hidden_state)
                scores = F.softmax(logits, dim=-1)[:, 1]
                distractor_scores[ready_for_cls_sample_inds] = scores
            gen_token_ids[:, i_step] = next_token_ids
            sample_lengths[(sample_lengths == 0) & (next_token_ids == self._eos_token_id)] = i_step + 1
            if torch.all(sample_lengths):
                break
            input_ids = next_token_ids.unsqueeze(1)
            past_key_values = model_out.past_key_values
            position_ids = position_ids[:, -1:] + 1

        sample_lengths[sample_lengths == 0] = i_step + 1
        return gen_token_ids, sample_lengths, distractor_scores


def _sample_next_token_ids(next_token_logits, top_k, temperature, token_ids_to_penalize, penalty):
    _penalize_specific_tokens(
        token_ids_to_penalize=token_ids_to_penalize,
        logits=next_token_logits,
        penalty=penalty,
    )
    next_token_logits.mul_(1 / temperature)
    top_k_logits, top_k_inds = torch.topk(next_token_logits, top_k)
    top_k_probas = F.softmax(top_k_logits, dim=-1)
    next_input_ids = torch.multinomial(top_k_probas.type(torch.float64), num_samples=1)
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


def _pad_left(arrays, pad_value):
    max_len = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), max_len), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, -len(a):] = a
    return out
