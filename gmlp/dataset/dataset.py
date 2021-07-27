from torch.utils.data import Dataset
import torch
import random
import json
import pandas as pd

class LoadDataset(Dataset):
    def __init__(self, corpus_path, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.corpus_path = corpus_path

        self.padding = 0
        self.mask = 4
        self.start = 2
        self.sep = 3

        with open("data/train_data/" + corpus_path, 'r') as js:
            self.dataset = json.load(js)
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item): # get bert input, label
        data = self.dataset[item]
        indice = data['indices']

        if self.seq_len < 512:
            indice = indice[:self.seq_len-1] + [self.sep]

        bert_input, bert_label = self.rand_dynamic_masking(indice)

        output = {"bert_input": bert_input, "bert_label": bert_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def rand_dynamic_masking(self, indice):
        output_label = []

        for i, token in enumerate(indice):
            org_token = indice[i]

            if org_token == self.start or org_token == self.sep:
                output_label.append(self.padding)
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    indice[i] = self.mask # mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    indice[i] = random.randrange(self.vocab_size) # vocab_size
                # 10% not change token

                if indice[i] == self.mask:
                    output_label.append(org_token)
                else:
                    output_label.append(self.padding)
            else:
                output_label.append(self.padding)

        return indice, output_label

class LoadDataset_nsmc(Dataset):
    def __init__(self, corpus_path, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.corpus_path = corpus_path

        self.padding = 0
        self.mask = 4
        self.start = 2
        self.sep = 3

        self.nsmc_dataset = []

        from transformers import ElectraTokenizer

        self.tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

        self.nsmc_dataset = pd.read_csv(corpus_path, sep='\t').dropna(axis=0)
        self.nsmc_dataset.drop_duplicates(subset=['document'], inplace=True)

        self.dataset_len = len(self.nsmc_dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        row = self.nsmc_dataset.iloc[item,1:3].values

        text = row[0]
        label = row[1]

        text = self.tokenizer.tokenize(text)

        if len(text) <= self.seq_len-2:
            text = [self.start] + self.tokenizer.convert_tokens_to_ids(text) + [self.sep]

            pad_length = self.seq_len - len(text)

            text += (pad_length * [self.padding])
        else:
            text = text[:self.seq_len-2]
            text = [self.start] + self.tokenizer.convert_tokens_to_ids(text) + [self.sep]

        model_input = text
        model_label = int(label)

        output = {"bert_input": model_input, "bert_label": model_label}

        return {key: torch.tensor(value) for key, value in output.items()}