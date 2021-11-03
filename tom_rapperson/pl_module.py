from pytorch_lightning import LightningModule
from transformers import GPT2LMHeadModel

from tom_rapperson.dataset import SerializedDataset


class PLModule(LightningModule):
    def __init__(self, huggingface_model_name, train_dir, valid_dir, batch_size, seed):
        super().__init__()
        self._huggingface_model_name = huggingface_model_name
        self._train_dir = train_dir
        self._valid_dir = valid_dir
        self._batch_size = batch_size
        self._seed = seed
        self._model = None

    def prepare_data(self):
        self._get_model()

    def setup(self, stage):
        self._model = self._get_model()

    def train_dataloader(self):
        return self._get_dataloader(self._train_dir)

    def val_dataloader(self):
        return self._get_dataloader(self._valid_dir)

    def _get_dataloader(self, dir_):
        dataset = SerializedDataset(dir_)
        dataloader = dataset.get_dataloader(
            batch_size=self._batch_size,
            seed=self._seed,
            is_distributed=True,
        )
        return dataloader

    def _get_model(self):
        return GPT2LMHeadModel.from_pretrained(self._huggingface_model_name)
