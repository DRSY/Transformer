import argparse
import time
from typing import Callable, Dict, List, Tuple
import copy
import os

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

    PAD = '<pad>'

    def __init__(self, mask_rate=0.15):
        super().__init__()
        self.mask_rate = mask_rate
        self.datas = self.generateFakedata()
        self.word2idx = {}
        self.idx2word = []
        self.wordcounter = {}

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
        self.idx2word.append(BERT_Trainer.PAD)
        for sentence in self.datas:
            sentence = sentence.strip()
            words = sentence.split()
            for word in words:
                self.wordcounter[word] = self.wordcounter.get(word, 0) + 1
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        print("build vocab done...")
        print("total distince words:{}".format(len(self.idx2word)))

    def build_dataset(self):
        class MyDataSet(data.Dataset):
            def __init__(self, datas):
                super().__init__()
                self.data = datas

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                raise NotImplementedError

    def train(self, model: nn.Module, optimizer, loss_function):
        """
            train the model
        """
        raise NotImplementedError


