import logging
import sys
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import MovieLensDataLoader
from model import GRU4Rec, MLMRecSys


def parse_args(args=None):
    parser = ArgumentParser()

    ## Required parameters for the model
    parser.add_argument('--model_type', default='BERT', type=str)

    ## Required parameters for data module
    parser.add_argument('--data_dir', default='MovieLens-1M', type=str)
    parser.add_argument('--tokenizer_name', default='bert-base-cased', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--min_session_length', default=3, type=int)
    parser.add_argument('--max_session_length', default=12, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    
    ## Required parameters for trainer module
    parser.add_argument('--gpus', default=-1, type=int)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--accelerator', default='ddp', type=str)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args(args)
    return args

def main(args):
    wandb_logger = pl.loggers.WandbLogger(project='LMRecSys')
    
    print(vars(args))
    pl.seed_everything(args.seed)

    dm = MovieLensDataLoader(
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_name_or_path=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        max_session_length=args.max_session_length,
        min_session_length=args.min_session_length,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup()

    if args.model_type == 'BERT':
        model = MLMRecSys.load_from_checkpoint(args.model_path)
    else:
        model = GRU4Rec.load_from_checkpoint(args.model_path)

    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision,
        logger=wandb_logger,
        accelerator=args.accelerator,
        # limit_val_batches=1, # TODO: uncomment for debugging
    )
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)