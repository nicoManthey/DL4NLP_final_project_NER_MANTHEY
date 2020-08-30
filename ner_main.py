# link to the paper:
# https://arxiv.org/pdf/1603.01360.pdf

# for tutorial on torchtext see:
# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

# link to download (large) glove vectors 50d (163MB) as txt:
# https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation

# link to download the CoNLL-2003 dataset which is used here:
# https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003

"""
To Do:
- make requirements file
"""

import argparse
from os import listdir
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
from math import sqrt
from operator import itemgetter
import matplotlib.pyplot as plt
from seaborn import heatmap

import pandas as pd
import numpy as np

import torchtext as tt
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TorchCRF import CRF

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def read_data(folder_path, file_name):
    """Read data from txt files, save as csv (to use for torchtext dataset)"""
    file_path = Path(folder_path).joinpath(file_name).absolute()
    print(file_path)
    words_ = []
    pos_ = []
    constituents_ = []
    bio_ = []

    sentences = []
    pos = []
    constituents = []
    bio = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if line == '-DOCSTART- -X- O O':
                continue
            if line == "\n":
                if words_ != []:
                    sentences.append(words_)
                    pos.append(pos_)
                    constituents.append(constituents_)
                    bio.append(bio_)
                words_ = []
                pos_ = []
                constituents_ = []
                bio_ = []
                continue
            if '-DOCSTART-' not in line and line != '\n':
                elems = line.split(' ')
                words_.append(elems[0])
                pos_.append(elems[1])
                constituents_.append(elems[2])
                bio_.append(elems[3].replace('\n', ''))

        dic = {}
        dic['sentences'] = sentences
        dic['pos'] = pos
        dic['constituents'] = constituents
        dic['bio'] = bio

        a = [str(' '.join(elem)) for elem in dic['sentences']]
        b = [str(' '.join(elem)) for elem in dic['pos']]
        c = [str(' '.join(elem)) for elem in dic['constituents']]
        d = [str(' '.join(elem)) for elem in dic['bio']]
        df = pd.DataFrame([a, b, c, d]).transpose()
        csv_path = str(folder_path + '/csv_' + file_name + '.csv')
        df.to_csv(csv_path, index=False, header=False)


class Dataset(object):
    """torchtext dataset which provides train, val and test iterators,
    loads the embeddings for each tokens and has easy-to-use 
    text_vocab and bio_vocab"""
    def __init__(self, folder_path, train_file, val_file, test_file, glove_file, batch_size):
        tokenizer = lambda x: x.split()
        TEXT = tt.data.Field(sequential=True, lower=True, tokenize=tokenizer)
        POS = tt.data.Field(sequential=True, lower=False)
        CONSTITUENTS = tt.data.Field(sequential=True, lower=False, unk_token=None)
        BIO = tt.data.Field(sequential=True, lower=False, unk_token=None)
        datafields = [("text", TEXT), ("pos", POS), ("constituents", CONSTITUENTS), ("bio", BIO)]

        train, val, test = TabularDataset.splits(path=folder_path,
                                                 train=train_file,
                                                 validation=val_file,
                                                 test=test_file, 
                                                 format='csv',
                                                 skip_header=False,
                                                 fields=datafields)

        TEXT.build_vocab(train, vectors=Vectors(glove_file))
        POS.build_vocab(train)
        CONSTITUENTS.build_vocab(train)
        BIO.build_vocab(train)

        self.text_vocab = TEXT.vocab
        self.bio_vocab = BIO.vocab

        self.bio_labels = [BIO.vocab.stoi[elem] for elem in BIO.vocab.itos \
                if elem not in ['<unk>', '<pad>']]

        self.word_embeddings = TEXT.vocab.vectors

        self.train_iterator = tt.data.BucketIterator(
            (train),
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator = tt.data.BucketIterator(
            (val),
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.test_iterator = tt.data.BucketIterator(
            (test),
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)


class Embeddings(nn.Module):
    '''
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    '''
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * sqrt(self.d_model)


class LSTM_Model(nn.Module):
    """LSTM Model with forward method"""
    def __init__(self,
                 embed_dim,
                 num_layers,
                 hidden_dim,
                 text_vocab,
                 bio_vocab):
        super(LSTM_Model, self).__init__()
        self.bio_vocab = bio_vocab
        self.text_vocab = text_vocab
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_dim = len(text_vocab)
        self.n_classes = len(bio_vocab)

        self.word_embeddings = Embeddings(self.embed_dim, self.vocab_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_classes)

    def print_name(self):
        """To use the model name in outputted performance graphics"""
        return self.__repr__().split('\n')[0][:-1]


    def forward(self, x):
        """Forward pass, returns probability dist over tags"""
        embeds = self.word_embeddings(x)
        embeds = self.dropout(embeds)
        lstm_out, c = self.lstm(embeds)
        lstm_out = lstm_out.view(-1, lstm_out.shape[2])
        tag_scores = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_scores, dim=1)


    def loss_fn(self, outputs, labels):
        """Mask out padded tokens and return loss"""
        labels = labels.view(-1)
        mask = (labels != self.bio_vocab.stoi['<pad>']).float()
        num_tokens = int(torch.sum(mask).item())
        outputs = outputs[range(outputs.shape[0]), labels]*mask
        return -torch.sum(outputs)/num_tokens


    def run_train_epoch(self, iterator, track_stats, epoch):
        """Train model for one epoch and record performance"""
        self.train()
        total_loss = 0
        all_y = []
        all_y_pred = []
        for i, batch in enumerate(iterator):
            text = batch.text
            bio = batch.bio
            self.zero_grad()
            tag_scores = self.forward(text)
            loss = self.loss_fn(tag_scores, bio)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            y = bio.view(1, -1).squeeze()
            y_pred = tag_scores.argmax(1).view(1, -1).squeeze()
            idl = (y != self.bio_vocab.stoi['<pad>']).nonzero().squeeze()
            y = list(itemgetter(*idl.tolist())(y.tolist()))
            y_pred = list(itemgetter(*idl.tolist())(y_pred.tolist()))
            all_y.extend(y)
            all_y_pred.extend(y_pred)
        track_stats.read_train_stats(all_y, all_y_pred, total_loss/100)


    def run_eval_epoch(self, iterator, track_stats, val_or_test):
        """Run the model in eval mode with val or test iterator and read_in performance"""
        self.eval()
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                text = batch.text
                bio = batch.bio
                tag_scores = self.forward(text)

                y = bio.view(1, -1).squeeze()
                y_pred = tag_scores.argmax(1).view(1, -1).squeeze()
                idl = (y != self.bio_vocab.stoi['<pad>']).nonzero().squeeze()
                y = list(itemgetter(*idl.tolist())(y.tolist()))
                y_pred = list(itemgetter(*idl.tolist())(y_pred.tolist()))
                all_y.extend(y)
                all_y_pred.extend(y_pred)
            if val_or_test == 'val':
                track_stats.read_val_stats(all_y, all_y_pred)
            if val_or_test == 'test':
                track_stats.read_test_stats(all_y, all_y_pred)


class LSTM_CRF_Model(nn.Module):
    """LSTM Model with CRF layer"""
    def __init__(self,
                 embed_dim,
                 num_layers,
                 hidden_dim,
                 text_vocab,
                 bio_vocab):
        super(LSTM_CRF_Model, self).__init__()
        self.bio_vocab = bio_vocab
        self.text_vocab = text_vocab
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_dim = len(text_vocab)
        self.n_classes = len(bio_vocab)

        self.word_embeddings = Embeddings(self.embed_dim, self.vocab_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_classes)
        self.crf = CRF(self.n_classes)


    def print_name(self):
        """To use the model name in outputted performance graphics"""
        return self.__repr__().split('\n')[0][:-1]


    def forward(self, x, bio):
        """Computes the forward pass and returns the loss and predicted tags"""
        mask = (x != self.text_vocab.stoi['<pad>'])
        embeds = self.word_embeddings(x)
        embeds = self.dropout(embeds)
        lstm_out, c = self.lstm(embeds)
        bz = lstm_out.shape[1]  # batch_size may vary in epoch's last iter
        seq_len = x.shape[0]
        lstm_out = lstm_out.view(-1, lstm_out.shape[2])
        tag_scores = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_scores, dim=1)
        num_tags = tag_scores.shape[1]
        emissions = tag_scores.view(seq_len, bz, num_tags)
        tags = torch.argmax(tag_scores, dim=1).view(seq_len, bz)
        loss = self.crf(emissions, bio, mask=mask)
        crf_tags = self.crf.decode(emissions)
        return -loss, crf_tags


    def run_train_epoch(self, iterator, track_stats, epoch):
        """Train model for one epoch and record performance"""
        self.train()
        total_loss = 0
        all_y = []
        all_y_pred = []
        for i, batch in enumerate(iterator):
            text = batch.text
            bio = batch.bio
            self.zero_grad()
            loss, crf_tags = self.forward(text, bio)
            crf_tags = torch.transpose(torch.IntTensor(crf_tags), 0, 1)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            y = bio.view(1, -1).squeeze()
            y_pred = crf_tags.reshape(1, -1).squeeze()
            idl = (y != self.bio_vocab.stoi['<pad>']).nonzero().squeeze()
            y = list(itemgetter(*idl.tolist())(y.tolist()))
            y_pred = list(itemgetter(*idl.tolist())(y_pred.tolist()))
            all_y.extend(y)
            all_y_pred.extend(y_pred)
        track_stats.read_train_stats(all_y, all_y_pred, total_loss/100)


    def run_eval_epoch(self, iterator, track_stats, val_or_test):
        """Run the model in eval mode with val or test iterator and read_in performance"""
        self.eval()
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                text = batch.text
                bio = batch.bio
                loss, crf_tags = self.forward(text, bio)
                crf_tags = torch.transpose(torch.IntTensor(crf_tags), 0, 1)

                y = bio.view(1, -1).squeeze()
                y_pred = crf_tags.reshape(1, -1).squeeze()
                idl = (y != self.bio_vocab.stoi['<pad>']).nonzero().squeeze()
                y = list(itemgetter(*idl.tolist())(y.tolist()))
                y_pred = list(itemgetter(*idl.tolist())(y_pred.tolist()))
                all_y.extend(y)
                all_y_pred.extend(y_pred)
            if val_or_test == 'val':
                track_stats.read_val_stats(all_y, all_y_pred)
            if val_or_test == 'test':
                track_stats.read_test_stats(all_y, all_y_pred)


class TrackStats():
    """A class to record, calculate and visualiye performance metrics"""
    def __init__(self, bio_labels, bio_vocab):
        self.bio_labels = bio_labels
        self.bio_vocab = bio_vocab
        ########
        self.train_loss = []
        self.train_acc = []
        self.train_f1_weighted = []
        self.train_f1_macro = []
        ########
        self.val_acc = []
        self.val_f1_weighted = []
        self.val_f1_macro = []
        ########
        self.test_acc = 0
        self.test_f1_weighted = 0
        self.test_f1_macro = 0
        ########
        self.train_all_y = None       # saves y and y_pred of current epoch
        self.train_all_y_pred = None
        self.val_all_y = None
        self.val_all_y_pred = None
        self.test_all_y = None
        self.test_all_y_pred = None

    def read_train_stats(self, all_y, all_y_pred, total_loss):
        """Use this after train iterator has run to calculate metrics"""
        self.train_loss.append(round(total_loss, 8))
        self.train_acc.append(round(accuracy_score(all_y, all_y_pred), 4))
        self.train_f1_weighted.append(round(f1_score(all_y, all_y_pred,\
                average='weighted', labels=self.bio_labels, zero_division=0), 4))
        self.train_f1_macro.append(round(f1_score(all_y, all_y_pred, \
                average='macro', labels=self.bio_labels, zero_division=0), 4))
        self.train_all_y = all_y
        self.train_all_y_pred = all_y_pred

    def read_val_stats(self, all_y, all_y_pred):
        """Use this after val iterator has run to calculate metrics"""
        self.val_acc.append(round(accuracy_score(all_y, all_y_pred), 4))
        self.val_f1_weighted.append(round(f1_score(all_y, all_y_pred,\
                average='weighted', labels=self.bio_labels, zero_division=0), 4))
        self.val_f1_macro.append(round(f1_score(all_y, all_y_pred,\
                average='macro', labels=self.bio_labels, zero_division=0), 4))
        self.val_all_y = all_y
        self.val_all_y_pred = all_y_pred

    def read_test_stats(self, all_y, all_y_pred):
        """Use this after test iterator has run to calculate metrics"""
        self.test_acc = round(accuracy_score(all_y, all_y_pred), 4)
        self.test_f1_weighted = round(f1_score(all_y, all_y_pred,\
                average='weighted', labels=self.bio_labels, zero_division=0), 4)
        self.test_f1_macro = round(f1_score(all_y, all_y_pred,\
                average='macro', labels=self.bio_labels, zero_division=0), 4)
        self.test_all_y = all_y
        self.test_all_y_pred = all_y_pred

    def print_train_stats(self, epoch):
        """Print train stats of current epoch"""
        print(f'Epoch: {epoch}')
        print(f'train acc: {self.train_acc[-1]:5.3f}\
                train F1 weighted: {self.train_f1_weighted[-1]:5.3f}\
                train F1 macro: {self.train_f1_macro[-1]:5.3f}\
                train loss: {self.train_loss[-1]:5.4f}')

    def print_val_stats(self):
        """Print val stats of current epoch"""
        print(f'val acc:   {self.val_acc[-1]:5.3f}\
                val F1 weighted:   {self.val_f1_weighted[-1]:5.3f}\
                val F1 macro:   {self.val_f1_macro[-1]:5.3f}')

    def train_loss_between_0_1(self):
        """Multiply train loss with a constant such that it is between 0 and 1
        (to be able to plot loss in a graph with acc), 
        used in method visualize()"""
        if self.train_loss[0] > 1:
            const_list = [10, 100, 1_000, 10_000, 100_000]
            for elem in const_list:
                if elem > self.train_loss[0]:
                    const = elem
                    break
        else:
            const = 1
        return [elem / const for elem in self.train_loss]

    def print_train_num_tag_occurences(self):
        """Print in terminal how often each tag was predicted in train epoch"""
        print('\nPredicted the following tags in this train epoch:')
        for elem in self.bio_labels:
            print(self.bio_vocab.itos[elem], self.train_all_y_pred.count(elem))

    def visualize(self, model_name, save_path=None):
        """Plots metrics over epochs for train, val and test
        and save to file"""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.plot(self.train_loss_between_0_1(), label='loss')
        ax1.plot(self.train_acc, label=f'acc: {self.train_acc[-1]:5.3f}')
        ax1.plot(self.train_f1_weighted, label=f'f1_weighted: {self.train_f1_weighted[-1]:5.3f}')
        ax1.plot(self.train_f1_macro, label=f'f1_macro: {self.train_f1_macro[-1]:5.3f}')
        ax1.set_yticks(ticks=np.arange(0, 1, 0.1))
        ax1.legend()
        ax1.grid()
        ax1.set(xlabel='Epoch')
        ax1.set_title(f'{model_name} / Train: loss, acc, f1')
        ax2.plot(self.val_acc, label=f'val_acc: {self.val_acc[-1]:5.3f}')
        ax2.plot(self.val_f1_weighted, label=f'val_f1_weighted: {self.val_f1_weighted[-1]:5.3f}')
        ax2.plot(self.val_f1_macro, label=f'val_f1_macro: {self.val_f1_macro[-1]:5.3f}')
        ax2.axhline(y=self.test_acc, xmin=0, xmax=len(self.train_acc), 
                    color='r', label=f'test_acc {self.test_acc:5.3f}')
        ax2.axhline(y=self.test_f1_weighted, xmin=0, xmax=len(self.train_acc), 
                    color='c', label=f'test_f1_weighted {self.test_f1_weighted:5.3f}')
        ax2.axhline(y=self.test_f1_macro, xmin=0, xmax=len(self.train_acc), 
                    color='m', label=f'test_f1_macro {self.test_f1_macro:5.3f}')
        ax2.set_yticks(ticks=np.arange(0, 1, 0.1))
        ax2.legend()
        ax2.grid()
        ax2.set(xlabel='Epoch')
        ax2.set_title(f'{model_name} / Val: acc, f1')
        fig.tight_layout(pad=1.2)
        #if save_path is not None:
        if args.save_matplotlib_graphs:
            plt.savefig(save_path)
        plt.show()

    def confusion_matrix(self, val_or_test, model_name, save_path=None, figsize=(8, 8)):
        """Calculates a confusion matrix for an iterator for one epoch
        and save to file"""
        if val_or_test == 'train':
            y_true = self.train_all_y
            y_pred = self.train_all_y_pred
        if val_or_test == 'val':
            y_true = self.val_all_y
            y_pred = self.val_all_y_pred
        if val_or_test == 'test':
            y_true = self.test_all_y
            y_pred = self.test_all_y_pred

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=np.unique(y_true))
        #cm_sum = np.sum(cm, axis=1, keepdims=True)
        #cm_perc = cm / cm_sum.astype(float) * 100

        data = [[str(self.bio_vocab.itos[label])+'\n'+str(sum(cm[l])), \
                str(self.bio_vocab.itos[pred])+'\n'+str(sum(cm[:, p])), \
                round(cm[l, p] / sum(cm[l]), 2)] \
            if pred in set(y_pred) \
            else [str(self.bio_vocab.itos[label])+'\n'+str(sum(cm[l])), \
                (str(self.bio_vocab.itos[pred]))+'\n'+str(0), 0] \
            for l, label in enumerate(sorted(set(y_true)))\
            for p, pred in enumerate(sorted(set(y_true)))]

        y_true_col = 'True\naboslute occurence of label'
        y_pred_col = 'Pred\naboslute occurence of predicted label'

        df = pd.DataFrame(data, columns=[y_true_col, y_pred_col, 'Value'])

        per_source_cat = df.groupby([y_true_col, y_pred_col]).agg({'Value': 'sum'})
        max_per_source = df.groupby([y_true_col]).agg({'Value': 'max'})
        per_source_cat = per_source_cat.div(max_per_source, level=y_true_col) * 100
        per_source_cat = per_source_cat.pivot_table(index=y_true_col, \
                columns=y_pred_col, values='Value')
        df = df.pivot_table(index=y_true_col, columns=y_pred_col, values='Value')
        fig, ax = plt.subplots(figsize=figsize)
        title_string = f'{model_name} / Confusion Matrix\n of {val_or_test} dataset'
        heatmap(per_source_cat, cmap='coolwarm', annot=df, fmt='g', \
                linewidths=1, linecolor='black', ).set_title(title_string)
        #if save_path is not None:
        if args.save_matplotlib_graphs:
            plt.savefig(save_path)
        plt.show()

    def classification_report_test(self, save_path):
        """Make classification report to get f1 score per label
        print in terminal and write to text file"""
        bio_labels_str = [self.bio_vocab.itos[elem] for elem in self.bio_labels]
        target_names = []
        for tag_id in self.test_all_y:
            if self.bio_vocab.itos[tag_id] not in target_names:
                target_names.append(self.bio_vocab.itos[tag_id])
        report = classification_report(self.test_all_y, self.test_all_y_pred, 
                                       target_names=bio_labels_str)
        with open(save_path, 'w') as file:
            file.write(report)
        print(classification_report(self.test_all_y, self.test_all_y_pred,
                                    target_names=bio_labels_str))



if __name__ == '__main__':

    # values used in paper written as comment in end of line
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=256) # 1
    parser.add_argument('--hidden_dim', type=int, default=100) # 100
    parser.add_argument('--n_lstm_layers', type=int, default=1) # 1
    parser.add_argument('--dropout', type=float, default=0.5) # 0.5
    parser.add_argument('--learning_rate', type=float, default=0.01) # 0.01
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--glove_file', 
                        default='glove.6B.50d.txt',
                        help='path to the glove vector file to use.')
    parser.add_argument('--save_matplotlib_graphs', type=bool, default=True,
                        help='if True, save matplotlib evaluation graphs to file')                    
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))
    print()

    ##############
    # Read in data
    ##############

    current_dir = str(Path(__file__).parent.absolute())
    folder_path = join(current_dir, 'CoNLL-2003')
    train_file = join(folder_path, 'csv_train.txt.csv')
    val_file = join(folder_path, 'csv_valid.txt.csv')
    test_file = join(folder_path, 'csv_test.txt.csv')
    glove_file = args.glove_file

    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) \
            and "csv" not in f and ".DS" not in f and "source" not in f]

    only_csv_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and "csv" in f]

    if len(only_csv_files) < 3:
        print('Data not converted to .csv format yet.\nStart converting to .csv now.')
        for file in onlyfiles:
            read_data(folder_path, file)

    ###########################
    # Instanciate class objects
    ###########################

    ds = Dataset(folder_path, train_file, val_file, test_file, glove_file, batch_size=args.batch_size)

    print('The following tags are included in the tag vocabulary:')
    for tag in ds.bio_vocab.stoi:
        print(tag)
    print()

    track_stats = TrackStats(ds.bio_labels, ds.bio_vocab)

    if args.use_crf:
        model = LSTM_CRF_Model(embed_dim=ds.word_embeddings.shape[1],
                            num_layers=args.n_lstm_layers,
                            hidden_dim=args.hidden_dim,
                            text_vocab=ds.text_vocab,
                            bio_vocab=ds.bio_vocab)

    if not args.use_crf:
        model = LSTM_Model(embed_dim=ds.word_embeddings.shape[1],
                        num_layers=args.n_lstm_layers,
                        hidden_dim=args.hidden_dim,
                        text_vocab=ds.text_vocab,
                        bio_vocab=ds.bio_vocab)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(model, '\n')

    #########################
    # Training and evaluation
    #########################

    print('The following is a progress bar over all epochs')
    for epoch in tqdm(range(1, 1 + args.epochs)):
        model.run_train_epoch(ds.train_iterator, track_stats, epoch)
        model.run_eval_epoch(ds.val_iterator, track_stats, 'val')
        #track_stats.print_train_num_tag_occurences()
        track_stats.print_train_stats(epoch)
        track_stats.print_val_stats()

    print("\nNow running test dataset.\n")
    model.run_eval_epoch(ds.val_iterator, track_stats, 'test')
    print(f"Test accuracy: {track_stats.test_acc:5.4f}    Test F1: {track_stats.test_f1_weighted:5.3f}")

    track_stats.visualize(model_name=model.print_name(), \
            save_path=join(current_dir, f'evaluation_graphics/F1_{model.print_name()}.png'))
    track_stats.confusion_matrix('train', model_name=model.print_name(), \
            save_path=join(current_dir, f'evaluation_graphics/train_CM_{model.print_name()}.png'))
    track_stats.confusion_matrix('test', model_name=model.print_name(), \
            save_path=join(current_dir, f'evaluation_graphics/test_CM_{model.print_name()}.png'))
    track_stats.classification_report_test(
            save_path=join(current_dir, f'{model.print_name()}_classification_report_test_set.txt'))
            