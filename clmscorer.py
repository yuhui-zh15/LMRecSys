import torch
import math
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import sys


class CausalLanguageModelScorer:

    def __init__(self, model_name='gpt2', device='cuda'):
        self.model_name = model_name
        self.device = device
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        if 'cuda' in self.device: self.model = self.model.to(self.device)

    def compute_loss(self, lm_logits, labels, mask):
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        # Flatten the tokens
        loss_matrix = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(-1, shift_labels.size(-1))
        loss_vec = (loss_matrix * shift_mask).sum(1) / shift_mask.sum(1)
        return loss_vec

    def score(self, sentence):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt")
            if 'cuda' in self.device: inputs = {key: inputs[key].to(self.device) for key in inputs}
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            return loss.item()

    def score_batch(self, sentences, max_length):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=max_length)
            if 'cuda' in self.device: inputs = {key: inputs[key].to(self.device) for key in inputs}
            outputs = self.model(**inputs)
            loss_vec = self.compute_loss(outputs.logits, inputs["input_ids"], inputs['attention_mask'])
            return loss_vec.tolist()


def read_jsonl(filename):
    return [json.loads(line) for line in open(filename)]


def save_jsonl(data, filename):
    with open(filename, 'w') as fout:
        for item in data:
            fout.write(json.dumps(item) + '\n')


def compute_acc(outputs):
    return np.mean([item['pred'] == item['answer'] for item in outputs])


def pattern1(ids, id2name):
    all_s = []
    s = 'A user watched '
    for id in ids:
        s += id2name[id] + ', '
    s = s.strip()[:-1] + '. '
    s += 'Now the user may want to watch '
    for id, name in enumerate(id2name):
        all_s.append(s + id2name[id] + '.')
    return all_s


def pattern2(ids, id2name):
    all_s = []
    s = ''
    for id in ids:
        s += id2name[id] + ', '
    for id, name in enumerate(id2name):
        all_s.append(s + id2name[id] + '.')
    return all_s


def pattern3(ids, id2name):
    all_s = []
    s = 'A user watched movies '
    for id in ids:
        s += id2name[id] + ', '
    s = s.strip()[:-1] + '. '
    s += 'Now the user may want to watch the movie '
    for id, name in enumerate(id2name):
        all_s.append(s + id2name[id] + '.')
    return all_s
    

def evaluate_lm(data_path, model_name, device, max_session_length):
    id2name = json.load(open(f'{data_path}/id2name.json'))
    data = [json.loads(line) for line in open(f'{data_path}/data.jsonl')]
    lm_scorer = CausalLanguageModelScorer(model_name, device=device)

    fout = open(model_name + '.jsonl', 'w')
    for item in tqdm(data):
        ids = item['ids'][:max_session_length]
        train_ids, val_id, test_id = ids[:-2], ids[-2], ids[-1]
        inputs = pattern(train_ids, id2name)
        scores = [lm_scorer.score(input) for input in tqdm(inputs)]
        fout.write(json.dumps(scores) + '\n')
    fout.close()


def evaluate_lm_batchify(data_path, model_name, device, max_session_length, max_seq_length, batch_size, pattern=1):
    id2name = json.load(open(f'{data_path}/id2name.json'))
    data = [json.loads(line) for line in open(f'{data_path}/data.jsonl')]
    lm_scorer = CausalLanguageModelScorer(model_name, device=device)

    fout = open(model_name + f'_pattern{pattern}_batchify.jsonl', 'w')
    for item in tqdm(data):
        ids = item['ids'][:max_session_length]
        train_ids, val_id, test_id = ids[:-2], ids[-2], ids[-1]
        inputs = eval(f'pattern{pattern}')(train_ids, id2name)
        scores = []
        for i in range(0, len(inputs), batch_size):
            scores += lm_scorer.score_batch(inputs[i: i + batch_size], max_length=max_seq_length)
        fout.write(json.dumps(scores) + '\n')
    fout.close()


def evaluate_lm_calibration(data_path, model_name, device, max_seq_length, batch_size, pattern=1):
    id2name = json.load(open(f'{data_path}/id2name.json'))
    lm_scorer = CausalLanguageModelScorer(model_name, device=device)

    inputs = eval(f'pattern{pattern}')([], id2name)
    scores = []
    for i in range(0, len(inputs), batch_size):
        scores += lm_scorer.score_batch(inputs[i: i + batch_size], max_length=max_seq_length)
    json.dump(scores, open(model_name + f'_pattern{pattern}_calibration.json', 'w'))
        
        
if __name__ == '__main__':
    evaluate_lm_batchify(data_path='datasets/MovieLens-1M', model_name=sys.argv[1], device='cuda', max_session_length=7, max_seq_length=128, batch_size=32, pattern=sys.argv[2])
    # evaluate_lm_calibration(data_path='datasets/MovieLens-1M', model_name=sys.argv[1], device='cuda', max_seq_length=128, batch_size=32, pattern=sys.argv[2])