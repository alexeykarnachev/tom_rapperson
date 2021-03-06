from itertools import chain
from pathlib import Path

import numpy as np
import torch
from more_itertools import chunked
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader, Dataset, Sampler

INPUT_IDS_FILE_NAME = 'input_ids'
SEQUENCE_LENGTHS_FILE_NAME = 'sequence_lengths'
PREFIX_LENGTHS_FILE_NAME = 'prefix_lengths'
POST_LENGTHS_FILE_NAME = 'post_lengths'


class SerializedDataset(Dataset):
    def __init__(self, dir_, distractor_p, end_of_prefix_token_id, end_of_target_token_id):
        dir_ = Path(dir_)
        self._distractor_p = distractor_p
        self._end_of_prefix_token_id = end_of_prefix_token_id
        self._end_of_target_token_id = end_of_target_token_id
        self._input_ids = np.load(dir_ / (INPUT_IDS_FILE_NAME + '.npy'))
        self._prefix_lengths = np.load(dir_ / (PREFIX_LENGTHS_FILE_NAME + '.npy'))
        self._sequence_lengths = np.load(dir_ / (SEQUENCE_LENGTHS_FILE_NAME + '.npy'))
        self._post_lengths = np.load(dir_ / (POST_LENGTHS_FILE_NAME + '.npy'))
        self._sequence_lengths_cumsum = np.cumsum(self._sequence_lengths)
        self._prefix_length_to_idx = {}
        for idx, length in enumerate(self._prefix_lengths):
            self._prefix_length_to_idx.setdefault(length, [])
            self._prefix_length_to_idx[length].append(idx)

    def __len__(self):
        return len(self._artist_ids)

    def __getitem__(self, idx):
        input_ids = self._get_input_ids(idx)

        is_distractor = self._distractor_p > np.random.rand()
        if is_distractor:
            prefix_length = self._prefix_lengths[idx]
            distractor_idx = np.random.choice(self._prefix_length_to_idx.get(prefix_length))
            distractor_input_ids = self._get_input_ids(distractor_idx)
            input_ids[:prefix_length] = distractor_input_ids[:prefix_length]

        post_length = torch.tensor(self._post_lengths[idx].astype(np.int32), dtype=torch.long)
        input_ids = torch.tensor(input_ids.astype(np.int64), dtype=torch.long)
        is_distractor = torch.tensor(is_distractor, dtype=torch.long)
        cls_token_pos = torch.where(input_ids == self._end_of_target_token_id)[0][0]
        return input_ids, post_length, cls_token_pos, is_distractor

    def _get_input_ids(self, idx):
        start_idx = 0 if idx == 0 else self._sequence_lengths_cumsum[idx - 1]
        end_idx = start_idx + self._sequence_lengths[idx]
        input_ids = self._input_ids[start_idx:end_idx]
        return input_ids

    def get_dataloader(self, batch_size, seed, samples_offset):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=_LengthSortSampler(
                lengths=self._sequence_lengths,
                sort_chunk_size=batch_size * 10,
                samples_offset=samples_offset,
                seed=seed,
                is_distributed=False,
            ),
            num_workers=24,
            collate_fn=_PaddingCollator(pad_value=0),
        )


def get_n_samples(dir_):
    sequence_lengths = np.load(dir_ / (SEQUENCE_LENGTHS_FILE_NAME + '.npy'))
    return len(sequence_lengths)


class _PaddingCollator:
    def __init__(self, pad_value):
        self._pad_value = pad_value

    def __call__(self, data_items):
        return [self._pad_vectors(vectors) for vectors in zip(*data_items)]

    def _pad_vectors(self, vectors):
        n_samples = len(vectors)
        if vectors[0].ndim == 0:
            tensor = torch.tensor(vectors, dtype=vectors[0].dtype)
        else:
            max_len = max(len(vector) for vector in vectors)
            tensor = torch.full((n_samples, max_len), fill_value=self._pad_value, dtype=vectors[0].dtype)
            for i, vector in enumerate(vectors):
                tensor[i, :len(vector)] = vector
        return tensor


class _LengthSortSampler(Sampler):
    def __init__(self, lengths, sort_chunk_size, samples_offset=0, seed=42, is_distributed=True):
        self._dataset_n_samples = len(lengths)
        self._current_offset = samples_offset % self._dataset_n_samples
        self._rank, self._world_size = _get_rank_and_world_size(is_distributed)

        self._worker_n_samples = _count_worker_n_samples(
            dataset_n_samples=self._dataset_n_samples,
            world_size=self._world_size,
            rank=self._rank,
        )

        self._inds = _get_length_sort_sampler_inds(lengths=lengths, sort_chunk_size=sort_chunk_size, seed=seed)

    def __iter__(self):
        start = self._current_offset + self._rank
        for i in range(start, self._dataset_n_samples, self._world_size):
            yield self._inds[i]
        self._current_offset = 0

    def __len__(self):
        return self._worker_n_samples


def _get_length_sort_sampler_inds(lengths, sort_chunk_size, seed):
    inds = np.argsort(lengths)
    largest_chunk = inds[-sort_chunk_size:][::-1]
    inds_chunks = list(chunked(inds[:-sort_chunk_size], sort_chunk_size))
    if seed is not None:
        np.random.RandomState(seed=seed).shuffle(inds_chunks)
    return list(chain(largest_chunk, *inds_chunks))


def _get_rank_and_world_size(is_distributed):
    if is_distributed:
        rank = get_rank()
        world_size = get_world_size()
    else:
        rank, world_size = 0, 1

    return rank, world_size


def _count_worker_n_samples(dataset_n_samples, world_size, rank):
    if rank >= world_size:
        raise ValueError('`rank` must be lower than `world_size`.')
    elif world_size > dataset_n_samples:
        raise ValueError('`world_size` must be lower or equal to `dataset_n_samples`.')
    worker_n_samples = dataset_n_samples // world_size
    remainder = (dataset_n_samples % world_size) > rank
    return worker_n_samples + remainder
