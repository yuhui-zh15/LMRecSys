import os
import json
import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial
import random


class MovieLensDataLoader(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str = 'datasets/MovieLens-1M',
        model_type: str = 'BERT',
        model_name_or_path: str = 'bert-base-cased',
        max_seq_length: int = 512,
        max_session_length: int = 12,
        min_session_length: int = 3,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        n_mask: int = 10,
        limit_n_train_data: int = -1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.max_session_length = max_session_length
        self.min_session_length = min_session_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_mask = n_mask
        self.limit_n_train_data = limit_n_train_data
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, no_cache=False):
        
        cached_features_dir = f'{self.data_dir}/cached_{self.tokenizer.name_or_path.replace("/", "#")}_{self.tokenizer.vocab_size}_{self.max_seq_length}_{self.min_session_length}_{self.max_session_length}_{self.n_mask}_{self.limit_n_train_data}'
        
        if os.path.exists(cached_features_dir) and not no_cache:
            print(f'Loading features from cached dir {cached_features_dir}')
            
            self.dataset = datasets.load_from_disk(cached_features_dir)
        else:
            print(f'Generating features to cached dir {cached_features_dir}')

            self.dataset = datasets.load_dataset('json', data_files={'all_data': f'{self.data_dir}/data.jsonl'})

            if self.limit_n_train_data != -1:
                random.seed(1234)
                idxs = random.sample(range(len(self.dataset['all_data'])), self.limit_n_train_data)
                print('Sampled idxs', idxs)
                self.dataset['train'] = self.dataset['all_data'].select(idxs)
                # self.dataset['val'] = self.dataset['all_data'].select(idxs)
                # self.dataset['test'] = self.dataset['all_data'].select(idxs)
            else:
                self.dataset['train'] = self.dataset['all_data']
                self.dataset['val'] = self.dataset['all_data']
                self.dataset['test'] = self.dataset['all_data']
            self.dataset.pop('all_data')
    
            self.id2name = json.load(open(f'{self.data_dir}/id2name.json'))
            # self.tokenizer.add_tokens(['_'.join(name.split()) for name in self.id2name])

            if self.model_type == 'BERT':
                self.dataset['train'] = self.dataset['train'].map(partial(self.convert_to_features, label_idxs=list(range(-55, -2))), batched=True, num_proc=8, batch_size=128) # TODO: list(range(-55, -2))
                self.dataset['val'] = self.dataset['val'].map(partial(self.convert_to_features, label_idxs=[-2]), batched=True, num_proc=8, batch_size=128)
                self.dataset['test'] = self.dataset['test'].map(partial(self.convert_to_features, label_idxs=[-1]), batched=True, num_proc=8, batch_size=128)
            else:
                self.dataset['train'] = self.dataset['train'].map(self.truncate, batched=True, num_proc=8, batch_size=128)
                self.dataset['val'] = self.dataset['val'].map(self.truncate, batched=True, num_proc=8, batch_size=128) 
                self.dataset['test'] = self.dataset['test'].map(self.truncate, batched=True, num_proc=8, batch_size=128)
            
            if not no_cache: self.dataset.save_to_disk(cached_features_dir)
        
        if self.model_type == 'BERT':
            self.dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'mask_idxs'])
        else:
            self.dataset.set_format(type='torch', columns=['ids'])
        print(self.dataset)

    def truncate(self, merged_examples):
        examples = [{key: merged_examples[key][i] for key in merged_examples} for i in range(len(merged_examples['ids']))]
        
        examples = [example for example in examples if len(example['ids']) >= self.min_session_length]
        for example in examples:
            example['ids'] = example['ids'][-self.max_session_length:]
        
        merged_results = {key: [example[key] for example in examples] for key in examples[0].keys()} if len(examples) != 0 else {}
        return merged_results

    def convert_to_features(self, merged_examples, label_idxs):
        
        def pattern(ids, id2name, mask_token, n_mask):
            s = 'A user watched '
            for id in ids:
                s += id2name[id] + ', '
            s = s.strip()[:-1] + '. '
            s += 'Now the user may want to watch'
            s += mask_token * n_mask
            s += '.'
            return s

        # def pattern(ids, id2name, mask_token, n_mask):
        #     s = 'A user watched '
        #     for id in ids:
        #         s += id2name[id] + ' ' + '_'.join(id2name[id].split()) + ', '
        #     s = s.strip()[:-1] + '. '
        #     s += 'Now the user may want to watch '
        #     s += mask_token 
        #     s += '.'
        #     return s

        # def pattern(ids, id2name, mask_token, n_mask):
        #     s = ''
        #     for id in ids:
        #         s += '_'.join(id2name[id].split()) + ' '
        #     s += mask_token
        #     return s
        
        examples = [{key: merged_examples[key][i] for key in merged_examples} for i in range(len(merged_examples['ids']))]
        
        examples = [example for example in examples if len(example['ids']) >= self.min_session_length]
        for example in examples:
            example['ids'] = example['ids'][-self.max_session_length:]
        
        results = []
        for example in examples:
            for label_idx in label_idxs:   
                l = len(example['ids'])
                if l + label_idx <= 0: continue
                input_str = pattern(example['ids'][:label_idx], self.id2name, self.tokenizer.mask_token, self.n_mask)

                result = self.tokenizer(
                    input_str,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=True
                )
                result['input_str'] = input_str
                result['ids'] = example['ids'][:label_idx]
                result['label'] = example['ids'][label_idx]
                result['mask_idxs'] = [i for i, id in enumerate(result['input_ids']) if id == self.tokenizer.mask_token_id]
                result['user'] = example['user']
                results.append(result)
            
        merged_results = {key: [result[key] for result in results] for key in results[0].keys()} if len(results) != 0 else {}
        return merged_results
        
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.eval_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)


if __name__ == '__main__':
    # print('Hello World!')
    dm = MovieLensDataLoader(        
        data_dir = 'datasets/MovieLens-1M-5Star',
        model_type = 'BERT',
        max_seq_length = 8,
        max_session_length = 4,
        min_session_length = 4,
        n_mask = 1
    )
    dm.setup()
    print(dm.dataset['train'][[0, -1]])
    # for batch in dm.train_dataloader():
    #     print([batch[key] for key in batch])
    #     print([(key, batch[key].shape) for key in batch])
    #     break
    