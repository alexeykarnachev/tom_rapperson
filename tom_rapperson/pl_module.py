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
            ul_alpha,
            distractor_p,
            n_accum_steps,
            warmup_ratio,
            seed,
    ):
        super().__init__()
        self._huggingface_model_name = huggingface_model_name
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._ul_alpha = ul_alpha
        self._distractor_p = distractor_p
        self._n_accum_steps = n_accum_steps
        self._warmup_ratio = warmup_ratio
        self._seed = seed
        data_dir = Path(data_dir)
        self._train_dir = data_dir / 'train'
        self._valid_dir = data_dir / 'valid'
        self._n_train_samples = get_n_samples(self._train_dir)
        self._encoder = SongsEncoder.load(data_dir)
        self._vocab_size = self._encoder.vocab_size
        self._model = None
        self._samples_offset = 0
        self.save_hyperparameters()

    def prepare_data(self):
        self._get_model()

    def setup(self, stage):
        self._model = self._get_model()
        self.hparams['gpt_config'] = json.dumps(self._model.backbone.config.__dict__, ensure_ascii=False)

    def train_dataloader(self):
        return self._get_dataloader(self._train_dir, samples_offset=self._samples_offset)

    def val_dataloader(self):
        return self._get_dataloader(self._valid_dir, samples_offset=0)

    def forward(self, batch):
        input_ids, post_lengths, cls_token_positions, is_distractor = batch
        lm_labels = input_ids.clone()
        for i, (post_length, is_distractor_) in enumerate(zip(post_lengths, is_distractor)):
            if is_distractor_:
                lm_labels[i, :] = -100
            else:
                lm_labels[i, :-post_length] = -100
        losses = self._model(
            input_ids=input_ids,
            is_distractor=is_distractor,
            lm_labels=lm_labels,
            cls_token_positions=cls_token_positions,
            ul_alpha=self._ul_alpha,
        )
        return losses

    def training_step(self, batch, batch_idx):
        loss, lm_loss, ul_loss, distratcor_loss = self.forward(batch)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('loss/train', loss, sync_dist=True, prog_bar=True)
        self.log('lm_loss/train', lm_loss, sync_dist=True, prog_bar=True)
        self.log('ul_loss/train', ul_loss, sync_dist=True, prog_bar=True)
        self.log('distractor_loss/train', distratcor_loss, sync_dist=True, prog_bar=True)
        self.log('learning_rate', current_lr)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, lm_loss, ul_loss, distratcor_loss = self.forward(batch)
        return loss.item(), lm_loss.item(), ul_loss.item(), distratcor_loss.item()

    def validation_epoch_end(self, outputs):
        losses, lm_losses, ul_losses, distractor_losses = zip(*outputs)
        self.log('loss/valid', np.mean(losses), sync_dist=True, prog_bar=True)
        self.log('lm_loss/valid', np.mean(lm_losses), sync_dist=True, prog_bar=True)
        self.log('ul_loss/valid', np.mean(ul_losses), sync_dist=True, prog_bar=True)
        self.log('distractor_loss/valid', np.mean(distractor_losses), sync_dist=True, prog_bar=True)

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

    def on_load_checkpoint(self, checkpoint):
        self._samples_offset = checkpoint['samples_offset']

    def on_save_checkpoint(self, checkpoint) -> None:
        world_size = self.trainer.world_size
        samples_offset = self._batch_size * world_size * checkpoint['global_step']
        checkpoint['samples_offset'] = samples_offset
        checkpoint['world_size'] = world_size
        checkpoint['epoch'] = self.trainer.current_epoch
        print(f'Data samples seen so far: {samples_offset}')

    def _get_dataloader(self, dir_, samples_offset):
        dataset = SerializedDataset(
            dir_=dir_,
            distractor_p=self._distractor_p,
            end_of_prefix_token_id=self._encoder.end_of_prefix_token_id,
            end_of_target_token_id=self._encoder.end_of_target_token_id,
        )
        dataloader = dataset.get_dataloader(
            batch_size=self._batch_size,
            seed=self._seed,
            samples_offset=samples_offset,
        )
        return dataloader

    def _get_model(self):
        model = get_model_from_huggingface_pretrained(
            model_name=self._huggingface_model_name,
            vocab_size=self._vocab_size,
        )
        return model
