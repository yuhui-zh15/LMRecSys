import logging
import sys
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import RecProbingDataLoader, RecProbingCLMDataLoader
from model import GPT, T5


def parse_args(args=None):
    parser = ArgumentParser()

    parser.add_argument('--data_dir', default='../datasets/Probing', type=str)
    parser.add_argument('--domain', default='books', type=str)
    parser.add_argument('--model_type', default='clm', type=str)
    parser.add_argument('--model_name_or_path', default='gpt2', type=str)
    parser.add_argument('--max_seq_length', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--gpus', default=-1, type=int)

    args = parser.parse_args(args)
    return args

def main(args):
    print(vars(args))
    wandb_logger = pl.loggers.WandbLogger(project='LMRecSys')
    
    dm = RecProbingDataLoader(
        data_dir=args.data_dir,
        domain=args.domain,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup()

    model = GPT(
        model_name_or_path=args.model_name_or_path,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        # accelerator=args.accelerator,
        limit_test_batches=200, # TODO: uncomment for debugging
    )
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)