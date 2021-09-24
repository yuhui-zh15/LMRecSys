import logging
import sys
import json
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import MovieLensDataLoader
from model import MLMRecSys, GRU4Rec


def parse_args(args=None):
    parser = ArgumentParser()

    ## Required parameters for the model
    parser.add_argument('--model_type', default='BERT', type=str)

    ## Required parameters for data module
    parser.add_argument('--data_dir', default='datasets/MovieLens-1M', type=str)
    parser.add_argument('--model_name_or_path', default='bert-base-cased', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--max_session_length', default=12, type=int)
    parser.add_argument('--min_session_length', default=3, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--n_mask', default=10, type=int)
    parser.add_argument('--limit_n_train_data', default=-1, type=int)
    parser.add_argument('--no_cache', default=0, type=int)

    ## Required parameters for model module
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--use_adapter', default=0, type=int)
    parser.add_argument('--loss_type', default='multiclass-hinge', type=str)
    
    ## Required parameters for trainer module
    parser.add_argument('--default_root_dir', default='.', type=str)
    parser.add_argument('--gpus', default=-1, type=int)
    parser.add_argument('--val_check_interval', default=0.5, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--gradient_clip_val', default=1.0, type=float)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--accelerator', default='ddp', type=str)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args(args)
    return args


def main(args):
    wandb_logger = pl.loggers.WandbLogger(project='LMRecSys', log_model=False)
    
    print(vars(args))
    pl.seed_everything(args.seed)

    dm = MovieLensDataLoader(
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        max_session_length=args.max_session_length,
        min_session_length=args.min_session_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        n_mask=args.n_mask,
        limit_n_train_data=args.limit_n_train_data,
    )
    dm.setup(no_cache=args.no_cache)

    if args.model_type == 'BERT':
        model = MLMRecSys(
            data_dir=args.data_dir,
            model_name_or_path=args.model_name_or_path,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            n_mask=args.n_mask,
            gpus=args.gpus,
            accumulate_grad_batches=args.accumulate_grad_batches,
            max_epochs=args.max_epochs,
            use_adapter=args.use_adapter,
            loss_type=args.loss_type,
        )
    else:
        model = GRU4Rec(
            rnn_type=args.model_type,
            learning_rate=args.learning_rate,
            nitem=len(json.load(open(f'{args.data_dir}/id2name.json')))
        )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor='val/r@20', mode='max')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        gpus=args.gpus,
        val_check_interval=args.val_check_interval,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=args.accelerator,
        logger=wandb_logger,
        # limit_train_batches=5, # TODO: uncomment for debugging
        # limit_val_batches=1, # TODO: uncomment for debugging
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)