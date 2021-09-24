import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AdapterConfig
)
from utils import compute_metrics, compute_precision_at_k, compute_recall_at_k, compute_mrr_at_k, compute_ndcg_at_k
import datetime


class MLMRecSys(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = 'bert-base-cased',
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        gpus: int = 1,
        accumulate_grad_batches: int = 1,
        max_epochs: int = 10,
        data_dir: str = 'datasets/MovieLens-1M',
        n_mask: int = 10,
        use_adapter: bool = False,
        loss_type: str = 'multiclass-hinge'
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config)
        # self.model = AutoModelForMaskedLM.from_config(config=self.config)
        self.create_verbalizers()
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # for n, p in self.model.named_parameters():
        #     for i in range(11):
        #         if f'layer.{i}' in n:
        #             p.requires_grad = False

        # if use_adapter:
        #     self.adapter_config = AdapterConfig.load('houlsby')
        #     self.model.add_adapter('rec', config=self.adapter_config)
        #     self.model.train_adapter('rec')
        #     self.model.set_active_adapters('rec')
        
    def create_verbalizers(self, init='mean-token'):
        id2name = json.load(open(f'{self.hparams.data_dir}/id2name.json'))
        id2tokens = self.tokenizer(id2name, add_special_tokens=False, padding='max_length', max_length=self.hparams.n_mask, truncation=True)['input_ids']
        self.label_idxs = torch.tensor(id2tokens)
        
        # id2concatname = ['_'.join(name.split()) for name in id2name]
       
        # self.tokenizer.add_tokens(id2concatname)
        # self.model.resize_token_embeddings(len(self.tokenizer)) 
        # self.label_idxs = self.tokenizer(id2concatname, add_special_tokens=False, return_tensors='pt', truncation=True, max_length=1)['input_ids']

        # if init == 'mean':
        #     print('Mean Init!!!')
        #     self.model.bert.embeddings.word_embeddings.weight.data[-len(id2name):] = self.model.bert.embeddings.word_embeddings.weight.data[:-len(id2name)].mean(0)
        # elif init == 'mean-token':
        #     print('Mean Token Init!!!')
        #     for name, idx in zip(id2name, self.label_idxs):
        #         idx = idx[0]
        #         tokens = self.tokenizer(name)['input_ids']
        #         self.model.bert.embeddings.word_embeddings.weight.data[idx] = self.model.bert.embeddings.word_embeddings.weight.data[tokens].mean(0)
        
        
    def forward(self, inputs):
        def batched_index_select_for_mask(x, idxs):
            # Here is the batched index selection for mask (check carefully!!!)
            # input: x (B x T x D), idxs (B x N)
            # output: y (B x N x D)
            # B = batch size, T = seq length, D = bert dim, N = number of masks
            B = x.size()[0]
            first_dim_idxs = torch.arange(B).unsqueeze(-1).type_as(idxs) 
            y = x[first_dim_idxs, idxs]
            return y
        
        def batched_index_select_for_label(x, idxs):
            # Here is the batched index selection for label (check carefully!!!)
            # input: x (B x N x D), idxs (K x N)
            # output: y (B x K x N)
            # B = batch size, N = number of masks, D = bert dim, K = number of labels
            N = idxs.size()[1]
            second_dim_idxs = torch.arange(N).unsqueeze(0).type_as(idxs)
            y = x[:, second_dim_idxs, idxs]
            return y
        
        def aggregate_for_label(x, idxs):
            # Here is the aggregation for label logits (check carefully!!!)
            # input: x (B x K x N), idxs (K x N)
            # output: y (B x K)
            # B = batch size, N = number of masks, K = number of labels
            # TODO: filter pad token
            return x.mean(-1)
            
        model_outputs = self.model(
            input_ids=inputs['input_ids'], 
            token_type_ids=inputs['token_type_ids'],
            attention_mask=inputs['attention_mask']
        )
        logits = model_outputs.logits # B x T x D
        if self.hparams.loss_type == 'multiclass-hinge': logits = self.log_softmax(logits) # B x T x D
        mask_logits = batched_index_select_for_mask(logits, inputs['mask_idxs']) # B x N x D, N = number of masks
        label_logits = batched_index_select_for_label(mask_logits, self.label_idxs) # B x K x N, K = number of labels
        label_logits_aggregated = aggregate_for_label(label_logits, self.label_idxs) # B x K
        # print(logits.shape, mask_logits.shape, label_logits.shape, label_logits_aggregated.shape)
        
        outputs = {
            'logits': logits,
            'mask_logits': mask_logits,
            'label_logits': label_logits,
            'label_logits_aggregated': label_logits_aggregated
        }
        return outputs
    
    def compute_loss(self, logits, labels):
        # Here computes multiclass hinge loss (check carefully!!!)
        # input: logits (B x K), labels (B)
        # output: loss (1)
        # B = batch size, K = number of labels

        if self.hparams.loss_type == 'multiclass-hinge':
            B = logits.size()[0]
            pos_probs = logits[torch.arange(B).type_as(labels), labels].unsqueeze(-1) # B x 1
            neg_probs = logits # B x K
            diffs = 1 + neg_probs - pos_probs # B x K
            diffs[diffs < 0] = 0 # B x K
            loss = diffs.sum(-1) # B
            loss = loss.mean(-1) # 1
        else:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
        return loss
            
    def training_step(self, batch, batch_idx):
        # print([(key, batch[key].shape) for key in batch], batch_idx)
        outputs = self(batch)
        logits = outputs['label_logits_aggregated']
        labels = batch['label']
        loss = self.compute_loss(logits, labels)
        result = compute_metrics((-logits).detach().cpu().numpy().argsort(-1), labels.detach().cpu().numpy(), prefix='train')
        result['train/loss'] = loss.item()
        self.log_dict(result)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs['label_logits_aggregated']
        labels = batch['label']
        loss = self.compute_loss(logits, labels)
        return {'loss': loss.item(), 'logits': logits.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        logits = np.concatenate([x['logits'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])
        loss = np.array([x['loss'] for x in outputs]).mean()
        result = compute_metrics((-logits).argsort(-1), labels, prefix='val')
        result['val/loss'] = loss
        self.log_dict(result, sync_dist=True)
        # self.write_prediction_dict({'logits': logits, 'labels': labels}, f'dumps/preds-val-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        logits = outputs['label_logits_aggregated']
        labels = batch['label']
        loss = self.compute_loss(logits, labels)
        return {'loss': loss.item(), 'logits': logits.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        logits = np.concatenate([x['logits'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])
        loss = np.array([x['loss'] for x in outputs]).mean()
        result = compute_metrics((-logits).argsort(-1), labels, prefix='test')
        result['test/loss'] = loss
        self.log_dict(result, sync_dist=True)
        # self.write_prediction_dict({'logits': logits, 'labels': labels}, f'dumps/preds-test-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')
        return loss
    
    def setup(self, stage):
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus))) # TODO: small bug for self.hparams.gpus when it is equal to -1
                // self.hparams.accumulate_grad_batches 
                * float(self.hparams.max_epochs)
            )
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class GRU4Rec(pl.LightningModule):
    def __init__(
        self,
        nitem: int, 
        rnn_type: str = 'GRU', 
        ninp: int = 128, 
        nhid: int = 128, 
        nlayers: int = 1, 
        dropout: float = 0.5, 
        tie_weights: bool = True,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nitem, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, nitem)
        self.criterion = nn.CrossEntropyLoss()

        if tie_weights:
            if nhid != ninp: raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        # input: (B x T)
        # output: (B x T x N)
        emb = self.drop(self.encoder(input)) # (B x T x D)
        output, _ = self.rnn(emb) # (B x T x D)
        output = self.drop(output) # (B x T x D)
        decoded = self.decoder(output) # (B x T x N)
        return decoded
            
    def training_step(self, batch, batch_idx):
        logits = self(batch['ids'][:, :-3]).reshape(-1, self.hparams.nitem)
        labels = batch['ids'][:, 1:-2].reshape(-1)
        loss = self.criterion(logits, labels)
        result = compute_metrics((-logits).detach().cpu().numpy().argsort(-1), labels.detach().cpu().numpy(), prefix='train')
        result['train/loss'] = loss.item()
        self.log_dict(result)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['ids'][:, :-2])[:, -1].reshape(-1, self.hparams.nitem)
        labels = batch['ids'][:, 1:-1][:, -1].reshape(-1)
        loss = self.criterion(logits, labels)
        return {'loss': loss.item(), 'logits': logits.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        logits = np.concatenate([x['logits'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])
        loss = np.array([x['loss'] for x in outputs]).mean()
        result = compute_metrics((-logits).argsort(-1), labels, prefix='val')
        result['val/loss'] = loss
        self.log_dict(result, sync_dist=True)
        # self.write_prediction_dict({'logits': logits, 'labels': labels}, f'dumps/preds-rnn-val-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch['ids'][:, :-1])[:, -1].reshape(-1, self.hparams.nitem)
        labels = batch['ids'][:, 1:][:, -1].reshape(-1)
        loss = self.criterion(logits, labels)
        return {'loss': loss.item(), 'logits': logits.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        logits = np.concatenate([x['logits'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])
        loss = np.array([x['loss'] for x in outputs]).mean()
        result = compute_metrics((-logits).argsort(-1), labels, prefix='test')
        result['test/loss'] = loss
        self.log_dict(result, sync_dist=True)
        # self.write_prediction_dict({'logits': logits, 'labels': labels}, f'dumps/preds-rnn-test-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

if __name__ == '__main__':
    model = MLMRecSys()