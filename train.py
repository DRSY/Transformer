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

log_step = 5


class Trainer:
    """
        The trainer class that helps to train a model for seq2seq purpose
        including methods:
                1.generate_fake_data
                2.build_vocab
                3.build_dataset
                4.build data loader
                5.train
                6.predict
    """

    SOS = '<sos>'
    EOS = '<eos>'
    PAD = '<pad>'
    src_MAXLEN = 0
    trg_MAXLEN = 0

    def __init__(self, epochs, batch_size, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.word2idx = {}
        self.idx2word = []
        self.wordcounter = {}
        self.trainning_set = None

        self.datas = Trainer.generateFakeData()
        self.build_vocab()
        self.trainning_set = self.buildDataset()
        self.train_loader = data.DataLoader(
            self.trainning_set, self.batch_size, True)

    @staticmethod
    def generateFakeData():
        """
            gnerate fake data
        """
        datas = [
            ["I love this class".split(), "Me too".split()],
            ["I hate this class".split(), "But why".split()],
            ["What do you like".split(), "I like eating".split()],
            ["What is your name".split(), "My name is roy".split()],
            ["Can you tell me your name".split(), "no I can not".split()],
            ["What time it is".split(), "I do not know".split()],
            ["What is your name".split(), "my name is roy".split()],
            ["Do you like this class".split(), "I do not like this class".split()],
            ["Do you like this shirt".split(), "I like this shirt".split()],
        ]
        return datas

    def build_vocab(self):
        """
            build vocabulary, compute src_MAXLEN and trg_MAXLEN for padding
        """
        print('build vocab begin')
        assert len(self.datas) > 1, "dataset is too small for trainning"
        self.word2idx[Trainer.PAD] = len(self.word2idx)
        self.word2idx[Trainer.EOS] = len(self.word2idx)
        self.word2idx[Trainer.SOS] = len(self.word2idx)
        self.idx2word.append(Trainer.PAD)
        self.idx2word.append(Trainer.EOS)
        self.idx2word.append(Trainer.SOS)
        for src, trg in self.datas:
            Trainer.src_MAXLEN = max(len(src), Trainer.src_MAXLEN)
            Trainer.trg_MAXLEN = max(len(trg), Trainer.trg_MAXLEN)
            for word in src+trg:
                word = word.lower()
                self.wordcounter[word] = self.wordcounter.get(word, 0) + 1
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        Trainer.trg_MAXLEN += 2
        print('build vocab done')
        print('total words:{}'.format(len(self.idx2word)))
        print("src_MAXLEN:{}, trg_MAXLEN:{}".format(
            Trainer.src_MAXLEN, Trainer.trg_MAXLEN))

    def buildDataset(self) -> data.Dataset:
        """
            build dataset
        """
        class MyDataSet(data.Dataset):
            def __init__(self, data, word2idx):
                super().__init__()
                self.data = data
                self.w2i = word2idx

            def __getitem__(self, index):
                """
                    pad src and trg sequence corresponing to their MAXLEN
                """
                data, target = copy.deepcopy(
                    self.data[index][0]), copy.deepcopy(self.data[index][1])
                while len(data) < Trainer.src_MAXLEN:
                    data.append(Trainer.PAD)
                target.insert(0, Trainer.SOS)
                target.append(Trainer.EOS)
                while len(target) < Trainer.trg_MAXLEN:
                    target.append(Trainer.PAD)
                data = list(map(lambda word: self.w2i[word.lower()], data))
                target = list(map(lambda word: self.w2i[word.lower()], target))
                return torch.tensor(data, dtype=torch.long), torch.tensor(target, dtype=torch.long)

            def __len__(self):
                return len(self.data)
        return MyDataSet(self.datas, self.word2idx)

    def initialize_model(self, model):
        """
            initialize model's params using xavier_uniform_
        """
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def train(self, model, optimizer, loss_function, early_stop=True):
        """
            train the model with given hyperparameters
            args:
                model: model to be trained
                optimizer: specific optimizer

            returns:
                loss_curve: 
        """
        self.initialize_model(model)
        model.train()
        print('trainning process begin')
        start = time.time()
        loss_curve = []
        for epoch in range(self.epochs):
            loss_ = .0
            for i, (src, trg) in enumerate(self.train_loader):
                optimizer.zero_grad()
                trg_input = trg[:, :-1]
                target = trg[:, 1:]
                src_mask, trg_mask = create_mask(src, trg_input)
                pred = model(src, trg_input, src_mask, trg_mask)
                # TODO:compute loss and backpropagate
                loss = loss_function(pred.transpose(-2, -1), target)
                loss_ += loss.item()
                loss.backward()
                optimizer.step()
            if self.verbose and epoch % log_step == 0:
                print("epoch:{}, loss:{}".format(
                    epoch+1, loss_/len(self.train_loader)))
            loss_curve.append(loss_/len(self.train_loader))
            if len(loss_curve) > 2 and loss_//len(self.train_loader) >= loss_curve[-1] >= loss_curve[-2]:
                print('early stop at epoch:{}'.format(epoch+1))
                break
        end = time.time()
        print('trainning process end')
        print('cost {} seconds'.format(end-start))
        return loss_curve

    def save_model(self, model: nn.Module):
        """
            save the model's structure and parameters
        """
        torch.save(model, "model.pkl")
        if os.path.exists("model.pkl"):
            print("save model successfully")
        else:
            print("model not saved due to some reason......")

    def evaluate(self):
        """
            evaluate the performence on test dataset
        """
        raise NotImplementedError

    def predict(self, model: nn.Module, src: str) -> str:
        """
            given input sentence, predict the corresponding output
        """
        model.eval()
        words = src.strip().split()
        if len(words) > Trainer.src_MAXLEN:
            print('input sentence is too long')
            return
        while len(words) < Trainer.src_MAXLEN:
            words.append(Trainer.PAD)
        words = torch.tensor(
            list(map(lambda word: self.word2idx[word.lower()], words)), dtype=torch.long)
        words = words.unsqueeze(0)  # (1, src_MAXLEN)
        src_mask = create_mask_src(words)
        e_output = model.encoder(words, src_mask)
        trg_inputidxs = [self.word2idx[Trainer.SOS]]
        trg_intput = torch.tensor(
            trg_inputidxs, dtype=torch.long).unsqueeze(0)  # (1, 1)
        while True:
            d_output = model.decoder(trg_intput, e_output, src_mask)
            output = model.output_layer(d_output)
            argmax_wordidx = torch.argmax(output[0][-1]).detach().item()
            if argmax_wordidx == self.word2idx[Trainer.EOS]:
                print('decoding up to EOS token...')
                break
            trg_inputidxs.append(argmax_wordidx)
            trg_intput = torch.tensor(
                trg_inputidxs, dtype=torch.long).unsqueeze(0)  # (1, new_length)
        decoded_output = ' '.join(
            list(map(lambda idx: self.idx2word[idx], trg_inputidxs[1:])))
        return decoded_output


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = Trainer(args.epochs, args.batch)
    model = Transformer(args.dmodel, len(trainer.idx2word),
                        args.heads, args.N, args.N, trainer.src_MAXLEN, trainer.trg_MAXLEN-1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    if args.train is False and os.path.exists('model.pkl'):
        model = torch.load("model.pkl")
    else:
        loss_curve = trainer.train(model, optimizer, loss_function)
        trainer.save_model(model)

    #plt.plot(range(len(loss_curve)), loss_curve)
    #plt.title('loss curve')
    # plt.show()
    print(trainer.predict(model, "what is your name"))
