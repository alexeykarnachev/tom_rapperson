import json
import math
from pathlib import Path

import numpy as np
from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup

from tom_rapperson.dataset import SerializedDataset, get_n_samples
from tom_rapperson.encoder import SongsEncoder
from tom_rapperson.model import get_model_from_huggingface_pretrained


class PLModule(LightningModule):
    def __init__(
            self,
            huggingface_model_name,
            data_dir,
            batch_size,
            learning_rate,
            n_accum_steps,
            warmup_ratio,
            seed,
    ):
        super().__init__()
        self._huggingface_model_name = huggingface_model_name
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._n_accum_steps = n_accum_steps
        self._warmup_ratio = warmup_ratio
        self._seed = seed
        data_dir = Path(data_dir)
        self._train_dir = data_dir / 'train'
        self._valid_dir = data_dir / 'valid'
        self._n_train_samples = get_n_samples(self._train_dir)
        encoder = SongsEncoder.load(data_dir)
        self._vocab_size = encoder.vocab_size
        self._model = None

    def prepare_data(self):
        self._get_model()

    def setup(self, stage):
        self._model = self._get_model()
        self.hparams['gpt_config'] = json.dumps(self._model.config.__dict__, ensure_ascii=False)

    def train_dataloader(self):
        return self._get_dataloader(self._train_dir)

    def val_dataloader(self):
        return self._get_dataloader(self._valid_dir)

    def forward(self, batch):
        input_ids = batch[0]
        output = self._model(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=input_ids != 0,
            return_dict=True,
        )
        return output.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('loss/train', loss, sync_dist=True, prog_bar=True)
        self.log('learning_rate', current_lr)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return loss.item()

    def validation_epoch_end(self, outputs):
        loss = np.mean(outputs)
        self.log('loss/valid', loss, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(params=self._model.parameters(), lr=self._learning_rate)
        batch_size_per_step = self._batch_size * self.trainer.world_size * self._n_accum_steps
        num_steps = self.trainer.max_epochs * math.ceil(self._n_train_samples / batch_size_per_step)
        num_warmup_steps = int(num_steps * self._warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps,
        )
        lr_scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _get_dataloader(self, dir_):
        dataset = SerializedDataset(dir_)
        dataloader = dataset.get_dataloader(batch_size=self._batch_size, seed=self._seed)
        return dataloader

    def _get_model(self):
        model = get_model_from_huggingface_pretrained(
            model_name=self._huggingface_model_name,
            vocab_size=self._vocab_size,
        )
        return model
