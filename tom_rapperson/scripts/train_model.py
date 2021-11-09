import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tom_rapperson.pl_module import PLModule


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--huggingface-model-name', required=True, type=str)
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--models-root-dir', required=True, type=str)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--learning-rate', required=True, type=float)
    parser.add_argument('--warmup-ratio', required=True, type=float)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--max-steps', required=True, type=int)
    parser.add_argument('--n-gpus', required=True, type=int)
    parser.add_argument('--val-check-interval', required=False, type=int, default=500)
    parser.add_argument('--model-name', required=False, type=str, default='tom_rapperson')
    return parser.parse_args()


def _calc_arguments_hash(*args):
    args = sorted(str(a) for a in args)
    return hashlib.md5(json.dumps(args).encode()).hexdigest()[:8]


def main(
        huggingface_model_name,
        data_dir,
        models_root_dir,
        batch_size,
        learning_rate,
        warmup_ratio,
        seed,
        max_steps,
        n_gpus,
        val_check_interval,
        model_name,
):
    seed_everything(seed)
    model_hash = _calc_arguments_hash(
        huggingface_model_name,
        batch_size * n_gpus,
        learning_rate,
        warmup_ratio,
        seed,
        max_steps,
    )
    model_dir = Path(models_root_dir) / model_hash
    model_dir.mkdir(exist_ok=True, parents=True)
    with open(model_dir / 'args.json', 'w') as out_file:
        json.dump(vars(args), out_file, indent=2)
    pl_module = PLModule(
        huggingface_model_name=huggingface_model_name,
        data_dir=data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        seed=seed,
    )
    logger = WandbLogger(
        name=model_name,
        id=model_hash,
        force=True,
        project='TomRapperson[RU]',
    )
    model_checkpoint = ModelCheckpoint(
        monitor='loss/valid',
        verbose=True,
        save_top_k=1,
        save_last=True,
        mode='min',
        dirpath=model_dir / 'checkpoints',
    )
    trainer = Trainer(
        gpus=n_gpus,
        replace_sampler_ddp=False,
        max_steps=max_steps,
        strategy='ddp' if args.n_gpus > 1 else None,
        precision=16,
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        weights_save_path=model_dir,
        callbacks=[model_checkpoint],
        logger=logger,
        val_check_interval=val_check_interval,
    )
    trainer.fit(pl_module)


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
