import argparse
import time
from typing import Callable, Dict, List, Tuple
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import f1_score

from myTransformer import *
from utils import create_mask, create_mask_src

parser = argparse.ArgumentParser()
parser.add_argument("--dmodel", type=int, default=512, required=True)
parser.add_argument("--heads", type=int, default=8, required=True)
parser.add_argument("--N", type=int, default=6, required=True)
parser.add_argument("--epochs", type=int, default=100, required=True)
parser.add_argument("--batch", type=int, default=64, required=True)
parser.add_argument("--train", type=bool, required=True)
args = parser.parse_args()


class BERT_Trainer:
    """
        The helper class to train the transformer model with 2 taskk
        1.masked language model
        2.next sentence prediction

        The pre-trained model can then be used in other downstream tasks with few changes to the model
        architecture
    """

    mask_rate = 0.15
    PAD = '<pad>'
    MASK = '<mask>'
    src_MAXLEN = 0

    def __init__(self, verbose=True):
        super().__init__()
        self.datas = self.generateFakedata()
        self.word2idx = {}
        self.idx2word = []
        self.wordcounter = {}
        self.epochs = args.epochs

        self.dataset = self.build_dataset(
            self.datas, self.idx2word, self.word2idx)
        self.loader = data.DataLoader(self.dataset, batch_size=2, shuffle=True)

    @staticmethod
    def getRandomNumber(self, low, high):
        """
            get random number in [low, high]
        """
        random.seed(1)
        n = random.randint(low, high)
        return n

    def generateFakedata(self):
        """
            generate fake data for masked language model task
        """
        datas = [
            "I like to shop ine morning",
            "There are a lot of beautiful things in the world",
            "Math may be the hardest subject in high school",
        ]
        return datas

    def mask(self):
        """
            mask the input sentence with certain probability
        """
        raise NotImplementedError

    def build_vocab(self):
        self.word2idx[BERT_Trainer.PAD] = len(self.word2idx)
        self.word2idx[BERT_Trainer.MASK] = len(self.word2idx)
        self.idx2word.append(BERT_Trainer.PAD)
        self.idx2word.append(BERT_Trainer.MASK)
        for sentence in self.datas:
            sentence = sentence.strip()
            words = sentence.split()
            BERT_Trainer.src_MAXLEN = max(BERT_Trainer.src_MAXLEN, len(words))
            for word in words:
                self.wordcounter[word] = self.wordcounter.get(word, 0) + 1
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        print("build vocab done...")
        print("total distince words:{}".format(len(self.idx2word)))
        print("src_MAXLEN:{}".format(BERT_Trainer.src_MAXLEN))

    def build_dataset(self):
        class MyDataSet(data.Dataset):
            def __init__(self, datas, idx2word, word2idx):
                super().__init__()
                self.data = datas
                self.idx2word = idx2word
                self.word2idx = word2idx

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                sentence = self.data[index]
                words = sentence.strip().split(" ")
                while len(words) < BERT_Trainer.src_MAXLEN:
                    words.append(BERT_Trainer.PAD)
                masking = [0] * BERT_Trainer.src_MAXLEN
                original_words = []
                for i, word in enumerate(words):
                    if word == BERT_Trainer.PAD:
                        break
                    n = BERT_Trainer.getRandomNumber(1, 100)
                    if n / 100 < BERT_Trainer.mask_rate:
                        m = BERT_Trainer.getRandomNumber(1, 100)
                        if m / 100 < 0.8:
                            """
                                replace the word with MASK
                            """
                            words[i] = BERT_Trainer.MASK
                            masking[i] = 1
                            original_words.append(word)
                        elif 0.9 > m / 100 >= 0.8:
                            """
                                replace the word with random words from vocab
                            """
                            while True:
                                words[i] = random.choice(self.idx2word)
                                if words[i] != BERT_Trainer.PAD:
                                    break
                        else:
                            """
                                let the word stay unchanged
                            """
                            pass
                datas = torch.tensor(
                    list(map(lambda word: self.word2idx[word.lower()], words)), dtype=torch.long)
                masking = torch.tensor(masking, dtype=torch.long)
                original_words = torch.tensor(
                    list(map(lambda word: self.word2idx[word.lower()])), dtype=torch.long)
                return datas, masking, original_words

        return MyDataSet(self.datas, self.idx2word, self.word2idx)

    def initialize_model(self, model):
        """
            initialize model's params using xavier_uniform_
        """
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def train(self, model: nn.Module, optimizer, loss_function):
        """
            train the model
        """
        model.train()
        start = time.time()
        print("train start...")
        for epoch in range(self.epochs):
            for i, (data, masking, original) in enumerate(self.loader):
                pass
        end = time.time()
        print("train done..")
        print("cost {}seconds".format(end-start))