import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
import os
import wget


# ==========================  dataset 형식 ==============================
class NSMCDataset(Dataset):

    def __init__(self, file_path, max_seq_len):
        self.data = pd.read_csv(file_path)
        self.max_seq_len = max_seq_len
        self.tokenizer = KoBERTTokenizer.from_pretrained('pretrained')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        doc = data['document']
        features = self.tokenizer.encode_plus(str(doc),       # 굉장히 편리하다. 잘 알아둘 것. 자세한 설명은 https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
                                              add_special_tokens=True,
                                              max_length=self.max_seq_len,
                                              pad_to_max_length='longest',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )
        input_ids = features['input_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)
        label = torch.tensor(data['label'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }

    # ==========================  전처리 ==============================
class NSMCDataModule(pl.LightningDataModule):

    def __init__(self, data_path, mode, valid_size, max_seq_len, batch_size):
        self.data_path = data_path
        self.full_data_path = f'{self.data_path}/train_{mode}.csv'
        self.test_data_path = f'{self.data_path}/test_{mode}.csv'
        self.valid_size = valid_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    #데이터 git 같은 곳에서 받아올 때 사용
    def prepare_data(self):
        # download data
        if not os.path.isfile(f'{self.data_path}/ratings_train.txt'):
            wget.download('https://github.com/e9t/nsmc/raw/master/ratings_train.txt', out=self.data_path)
        if not os.path.isfile(f'{self.data_path}/ratings_test.txt'):
            wget.download('https://github.com/e9t/nsmc/raw/master/ratings_test.txt', out=self.data_path)
        generate_preprocessed(self.data_path)

    def setup(self, stage):
        if stage in (None, 'fit'):
            full = NSMCDataset(self.full_data_path, self.max_seq_len)
            train_size = int(len(full) * (1 - self.valid_size))
            valid_size = len(full) - train_size
            self.train, self.valid = random_split(full, [train_size, valid_size])

        elif stage in (None, 'test'):
            self.test = NSMCDataset(self.test_data_path, self.max_seq_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=5, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=5, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=5, shuffle=False, pin_memory=True)