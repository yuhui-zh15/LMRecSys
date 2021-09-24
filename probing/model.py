import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from utils import compute_metrics, compute_precision_at_k, compute_recall_at_k, compute_mrr_at_k, compute_ndcg_at_k
import datetime


class T5(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = 't5-base',
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):

        def compute_log_probs(logits, labels, mask):
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            # Flatten the tokens
            loss_matrix = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(-1, shift_labels.size(-1))
            loss_vec = loss_matrix.mean(-1) # TODO: mean or sum
            return -loss_vec

        model_outputs = self.model(
            input_ids=inputs['input_ids'],
            labels=inputs['output_ids'],
        )

        # answer_log_probs = compute_log_probs(model_outputs.logits, inputs['output_ids'])

        outputs = {
            'answer_log_probs': torch.tensor([-model_outputs.loss]) # answer_log_probs
        }
        return outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        log_probs = outputs['answer_log_probs']
        labels = batch['label']
        return {'log_probs': log_probs.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        log_probs = np.concatenate([x['log_probs'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])

        acc = (log_probs.reshape(-1, 2).argmax(-1) == 0).mean()
        self.log_dict({'acc': acc}, sync_dist=True)
        self.write_prediction_dict({'log_probs': log_probs, 'labels': labels}, f'dumps/preds-test-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')


class GPT(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = 'gpt2',
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=self.config)

        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):

        def compute_log_probs(logits, labels, mask):
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            # Flatten the tokens
            loss_matrix = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(-1, shift_labels.size(-1))
            loss_vec = (loss_matrix * shift_mask).sum(1) / shift_mask.sum(1) # TODO: mean or sum
            return -loss_vec

        model_outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        answer_log_probs = compute_log_probs(model_outputs.logits, inputs['input_ids'], inputs['answer_mask'])

        outputs = {
            'answer_log_probs': answer_log_probs
        }
        return outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        log_probs = outputs['answer_log_probs']
        labels = batch['label']
        return {'log_probs': log_probs.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        log_probs = np.concatenate([x['log_probs'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])

        acc = (log_probs.reshape(-1, 2).argmax(-1) == 0).mean()
        self.log_dict({'acc': acc}, sync_dist=True)
        self.write_prediction_dict({'log_probs': log_probs, 'labels': labels}, f'dumps/preds-test-{datetime.datetime.today().strftime("%Y%m%d%H%M%S")}.pt')
