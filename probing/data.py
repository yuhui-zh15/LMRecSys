import os
import json
import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer
from functools import partial
import random


class RecProbingDataLoader(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str = '../datasets/Probing',
        domain: str = 'books',
        model_type: str = 'clm',
        model_name_or_path: str = 'gpt2',
        max_seq_length: int = 64,
        eval_batch_size: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.domain = domain
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token # TODO: no pad token for gpt2 tokenizer

    def setup(self, no_cache=False):
        
        self.dataset = datasets.load_dataset('csv', data_files={'test': f'{self.data_dir}/{self.domain}.csv'})
        self.dataset = self.dataset.map(self.convert_to_features, batched=True, num_proc=32, batch_size=128, remove_columns=['query', 'relevant_doc'] + [f'non_relevant_{i}' for i in range(1, 51)])
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer_mask', 'label']) # 'token_type_ids'

        print(self.dataset)


    def convert_to_features(self, merged_examples):

        def span_include(src, tgt):
            return src[0] <= tgt[0] and tgt[1] <= src[1]
        
        examples = [{key: merged_examples[key][i] for key in merged_examples} for i in range(len(merged_examples['query']))]
        
        results = []
        for example in examples:
            try:
                query = example['query'].split(' [SEP] ')[0].strip()
                pos_doc = example['relevant_doc'].strip()
                neg_doc = example['non_relevant_1'].strip()
            except:
                continue

            input_str = f'If you liked {query}, you will like {pos_doc} more than {neg_doc}.'
            result = self.tokenizer(
                input_str,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True
            )
            result['input_str'] = input_str
            result['label'] = 1
            result['answer_mask'] = [1 for i, token_span in enumerate(result['offset_mapping'])]
            results.append(result)

            input_str = f'If you liked {query}, you will like {neg_doc} more than {pos_doc}.'
            result = self.tokenizer(
                input_str,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True
            )
            result['input_str'] = input_str
            result['label'] = 0
            result['answer_mask'] = [1 for i, token_span in enumerate(result['offset_mapping'])]
            results.append(result)


            # for idx, doc in enumerate([pos_doc, neg_doc]):
            #     doc_, query_ = query, doc # doc, query # 
            #     input_str = f'If you liked {query_}, you will also like {doc_}.'
            #     result = self.tokenizer(
            #         input_str,
            #         max_length=self.max_seq_length,
            #         padding='max_length',
            #         truncation=True,
            #         return_offsets_mapping=True
            #     )
            #     span = (input_str.find(doc_) - 1, input_str.find(doc_) + len(doc_)) # TODO: -1 because GPT tokenizer use space before token

            #     result['input_str'] = input_str
            #     result['label'] = 1 if idx == 0 else 0
            #     result['answer_mask'] = [int(span_include(span, token_span)) for i, token_span in enumerate(result['offset_mapping'])]
            #     results.append(result)
            
        merged_results = {key: [result[key] for result in results] for key in results[0].keys()} if len(results) != 0 else {}
        return merged_results
    
    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)


class RecProbingCLMDataLoader(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str = '../datasets/Probing',
        domain: str = 'books',
        model_type: str = 'clm',
        model_name_or_path: str = 't5-base',
        max_seq_length: int = 64,
        eval_batch_size: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.domain = domain
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.eval_batch_size = eval_batch_size

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token # TODO: no pad token for gpt2 tokenizer

    def setup(self, no_cache=False):
        
        self.dataset = datasets.load_dataset('csv', data_files={'test': f'{self.data_dir}/{self.domain}.csv'})
        self.dataset = self.dataset.map(self.convert_to_features, batched=True, num_proc=32, batch_size=128, remove_columns=['query', 'relevant_doc'] + [f'non_relevant_{i}' for i in range(1, 51)])
        self.dataset.set_format(type='torch', columns=['input_ids', 'output_ids', 'label']) # 'token_type_ids'

        print(self.dataset)


    def convert_to_features(self, merged_examples):

        def span_include(src, tgt):
            return src[0] <= tgt[0] and tgt[1] <= src[1]
        
        examples = [{key: merged_examples[key][i] for key in merged_examples} for i in range(len(merged_examples['query']))]
        
        results = []
        for example in examples:

            query = example['query'].split(' [SEP] ')[0].strip()
            pos_doc = example['relevant_doc'].strip()
            neg_doc = example['non_relevant_1'].strip()

            for idx, doc in enumerate([pos_doc, neg_doc]):
                doc_, query_ = query, doc # doc, query # 

                input_str = f'If you liked {query_}, you will also like <extra_id_0>.'
                input_ids = self.tokenizer(input_str).input_ids
                output_str = f'<extra_id_0> {doc_} <extra_id_1> </s>'
                output_ids = self.tokenizer(output_str).input_ids

                # span = (input_str.find(doc_) - 1, input_str.find(doc_) + len(doc_)) # TODO: -1 because GPT tokenizer use space before token

                result = {
                    'input_ids': input_ids,
                    'output_ids': output_ids,
                    'label': 1 if idx == 0 else 0
                }
                results.append(result)
            
        merged_results = {key: [result[key] for result in results] for key in results[0].keys()} if len(results) != 0 else {}
        return merged_results
    
    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)


if __name__ == '__main__':
    dm = RecProbingDataLoader()
    dm.setup()
    print(dm.dataset['test'][0])
    print(dm.dataset['test'][1])
    # for batch in dm.train_dataloader():
    #     print([batch[key] for key in batch])
    #     print([(key, batch[key].shape) for key in batch])
    #     break
    