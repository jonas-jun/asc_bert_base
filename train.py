import argparse
import os, sys
from data_utils import Category_Classification_Dataset as Dataset
from data_utils import clean_sentence, preprocess
from transformers import BertTokenizer
from data_utils import custom_random_split as rs
from torch.utils.data import DataLoader
# models
from models.bert_intermediate import *
from models.bert_attscores import Bert_AttScore, Bert_AttScore_RNN, Bert_AttScore_RNN_add

# train
import torch.nn as nn
import torch.optim as optim
from custom_trainer import evaluate, trainer, runs





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model.name', type=str)
    parser.add_argument('--dataset_series', default='SemEval-16')
    parser.add_argument('--dataset_domain', default='restaurant')
    


    opt = parser.parse_args()


    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




















if __name__ == '__main__':
    main()    






# for insert mode