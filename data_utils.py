from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import nltk
import re
from nltk import pos_tag
from nltk import RegexpParser
import random
import numpy as np
from torch.utils.data import random_split
import os

class Term_Classification_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length, relative_pos_enc=False, pair=True):
        self.df = df
        self.tokenizer = tokenizer
        self.dataset = list()
        self.asp_idxs = list()
        self.relative_pos_enc = relative_pos_enc

        list_sentence = df.sentence
        list_term = df.target
        list_polarity = df.polarity
        list_category = df.category

        self.label_map = {'negative': 0, 'positive': 1, 'neutral': 2, 'conflict': 3}

        print('{:,} samples in this dataset'.format(len(df)))

        for sentence, term, polarity in zip(list_sentence, list_term, list_polarity):
            if pair:
                encoded = tokenizer.encode_plus(text=sentence, text_pair=term, add_special_tokens=True,
                    padding='max_length', max_length=max_length, pad_to_max_length=True,
                    return_token_type_ids=True, return_tensors='pt')
            else:
                encoded = tokenizer.encode_plus(text=sentence, add_special_tokens=True,
                    padding='max_length', max_length=max_length, pad_to_max_length=True,
                    return_token_type_ids=True, return_tensors='pt')

            # for relative position encoding
            if relative_pos_enc:
                r_pos_enc = relative_position_encoding(sentence, term, self.tokenizer, max_length)
                # for last <SEP> token idx
                last_idx = (encoded['input_ids']==102).nonzero(as_tuple=True)[-1][-1].item()

                # for aspect words idx
                # asp_idx = (r_pos_enc==0).nonzero(as_tuple=True)[0].tolist()
                # self.asp_idxs.append(asp_idx)

                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'],
                    'token_type_ids': encoded['token_type_ids'], 'labels': self.label_map[polarity], 'pos': r_pos_enc,
                    'last_ids': last_idx} 
            else:
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                        'labels': self.label_map[polarity]}

            self.dataset.append(data)
        
    def get_sample(self, idx):
        print('Sentence: {}'.format(self.df.sentence[idx]))
        print('Aspect Term: {}'.format(self.df.target[idx]))
        print('Polarity: {}'.format(self.df.polarity[idx]))
        print('Input IDs: {}'.format(self.dataset[idx]['input_ids']))
        print('Token type IDs: {}'.format(self.dataset[idx]['token_type_ids']))
        print('Encoded Label: {}'.format(self.dataset[idx]['labels']))
        if self.relative_pos_enc:
            print('Relative Pos Enc: {}'.format(self.dataset[idx]['pos']))

    def get_asp_idxs(self):
        return self.asp_idxs
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class Term_Classification_Dataset_add_special(Term_Classification_Dataset):
    def __init__(self, df, tokenizer, max_length, relative_pos_enc=False, pair=True, where='before'):
        assert where in ['both', 'left', 'before', 'right', 'next']
        self.df = df
        if where == 'both':
            s_tokens = ['[ASP1]', '[ASP2]']
        elif where in ['left', 'before', 'right', 'next']:
            s_tokens = ['[ASP1]']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
            additional_special_tokens=s_tokens)
        self.dataset = list()
        self.asp_idxs = list()
        self.relative_pos_enc = relative_pos_enc

        list_sentence = df.sentence
        list_term = df.target
        list_polarity = df.polarity
        list_category = df.category
        list_start = df['from']
        list_end = df['to']

        # add special word [ASP1], [ASP2] each 30522, 30523 idx in vocab
        list_sentence_new = list()
        for text, from_, to_ in zip(list_sentence, list_start, list_end):
            text_left, text_center, text_right = text[:from_], text[from_:to_], text[to_:]
            if where == 'both':
                new = text_left + '[ASP1] ' + text_center + ' [ASP2]' + text_right
            elif where in ['left', 'before']:
                new = text_left + '[ASP1] ' + text_center + text_right
            elif where in ['right', 'next']:
                new = text_left + text_center + ' [ASP1]' + text_right
            list_sentence_new.append(new)

        self.label_map = {'negative': 0, 'positive': 1, 'neutral': 2, 'conflict': 3}

        print('{:,} samples in this dataset'.format(len(df)))

        for sentence, term, polarity in zip(list_sentence_new, list_term, list_polarity):
            if pair:
                encoded = tokenizer.encode_plus(text=sentence, text_pair=term, add_special_tokens=True,
                    padding='max_length', max_length=max_length, pad_to_max_length=True,
                    return_token_type_ids=True, return_tensors='pt')
            else:
                encoded = tokenizer.encode_plus(text=sentence, add_special_tokens=True,
                    padding='max_length', max_length=max_length, pad_to_max_length=True,
                    return_token_type_ids=True, return_tensors='pt')
            
            # for special token location
            special_locs = list()
            if where == 'both':
                special_locs.append((encoded['input_ids']==30522).nonzero(as_tuple=True)[-1].item())
                special_locs.append((encoded['input_ids']==30523).nonzero(as_tuple=True)[-1].item())
            if where in ['left', 'before', 'right', 'next']:
                special_locs.append((encoded['input_ids']==30522).nonzero(as_tuple=True)[-1].item())

            # for relative position encoding
            if relative_pos_enc:
                r_pos_enc = relative_position_encoding(sentence, term, tokenizer, max_length)
                # for last <SEP> token idx
                last_idx = (encoded['input_ids']==102).nonzero(as_tuple=True)[-1][-1].item()

                # for aspect words idx
                # asp_idx = (r_pos_enc==0).nonzero(as_tuple=True)[0].tolist()
                # self.asp_idxs.append(asp_idx)

                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'],
                    'token_type_ids': encoded['token_type_ids'], 'labels': self.label_map[polarity], 'pos': r_pos_enc,
                    'last_ids': last_idx, 'special_tokens': special_locs} 
            else:
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                        'labels': self.label_map[polarity], 'special_tokens': special_locs}

            self.dataset.append(data)

class Category_Classification_Dataset(Dataset):
    def __init__(self, df, tokenizer, opt, pos_encoding=False,
            pos_idx_zero=['[UNK]', '[SEP]', '[CLS]', '[PAD]', ',', '.']):
        self.sentihood = True if opt.dataset_series == 'sentihood' else False
        print('sentihood: {}'.format(self.sentihood))
        self.tokenizer = tokenizer
        if self.sentihood:
            self.df, _ = sum_category(df)
            # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
            #     additional_special_tokens=['LOCATION1', 'LOCATION2'])
        else:
            self.df = df
            self.tokenizer = tokenizer
        self.dataset = list()
        self.pos_vocab = dict()
        self.pos_encoding = pos_encoding
        if pos_encoding:
            nltk.download('averaged_perceptron_tagger')
        list_sentence = df.sentence
        list_polarity = df.polarity
        list_category = df.category

        self.label_map = {'negative': 0, 'positive': 1, 'neutral': 2, 'conflict': 3}

        print('{:,} samples in this dataset'.format(len(df)))
        
        i = 1
        for sentence, category, polarity in zip(list_sentence, list_category, list_polarity):
            category_words = ' '.join(category.split('#'))
            encoded = self.tokenizer.encode_plus(text=sentence, text_pair=category_words, add_special_tokens=True,
                    padding='max_length', max_length=opt.max_length, pad_to_max_length=True,
                    return_token_type_ids=True, return_tensors='pt')
            
            if pos_encoding:
                tag_pair = pos_tag(self.tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze(0)))
                list_pos = list()
                for word, pos in tag_pair:
                    if word in pos_idx_zero:
                        self.pos_vocab[word] = 0
                        list_pos.append(word)
                    else:
                        if pos in self.pos_vocab.keys():
                            list_pos.append(pos)
                        else:
                            self.pos_vocab[pos] = i
                            i += 1
                            list_pos.append(pos)
                encoded_pos = [self.pos_vocab[tag] for tag in list_pos]
                tensor_pos = torch.tensor(encoded_pos, dtype=torch.int8)
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                    'labels': self.label_map[polarity], 'pos': tensor_pos} 
            
            else:
                data = {'input_ids': encoded['input_ids'], 'attention_masks': encoded['attention_mask'], 'token_type_ids': encoded['token_type_ids'],
                        'labels': self.label_map[polarity]}
            
            self.dataset.append(data)

    def get_sample(self, idx):
        print('Sentence: {}'.format(self.df.sentence[idx]))
        print('Aspect Category: {}'.format(self.df.category[idx]))
        print('Polarity: {}'.format(self.df.polarity[idx]))
        print('Input IDs: {}'.format(self.dataset[idx]['input_ids']))
        print('Token type IDs: {}'.format(self.dataset[idx]['token_type_ids']))
        print('Encoded Label: {}'.format(self.dataset[idx]['labels']))
        if self.pos_encoding:
            print('Pos_encoding: {}'.format(self.dataset[idx]['pos']))

    def get_pos_vocab(self):
        return self.pos_vocab

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

def sum_category(df):
    '''
    for sentihood dataset
    category: price, target: LOCATION1 to category: LOCATION1#price

    @args
    df: data frame to apply
    '''
    df_old = df.copy()
    new = [df['target'][i]+'#'+df['category'][i] for i in range(len(df))]
    df['category'] = new
    return df, df_old

def relative_position_encoding(sentence, target, tokenizer, max_length):
    sent = tokenizer.tokenize(sentence)
    term = tokenizer.tokenize(target)
    sent_length = len(sent)
    term_length = len(term)

    sent_space = ' '.join(sent)
    term_space = ' '.join(term)
    temp_idx = sent_space.index(term_space)
    from_idx = len(sent_space[:temp_idx-1].split()) if temp_idx != 0 else 0
    to_idx = from_idx + term_length

    center = [0] * term_length
    first_cls = [max_length]
    left = list(range(from_idx, 0, -1)) # [5, 4, 3, 2, 1]
    right = list(range(1, sent_length-to_idx+1)) # [1, 2, 3, 4, 5, 6, ...]
    pad = [max_length] * (max_length-sent_length-1)
    return torch.tensor(first_cls + left + center + right + pad, dtype=torch.int16)

def reverse_pos(dataset, max_length):
    '''
    @params
    dataset: trainset or testset to change
    max_length: max sequence length, opt.max_length
    @return
    reversed dataset
    '''
    for data in dataset:
        data['pos'] = max_length - data['pos']
    return dataset

def custom_random_split(dataset, val_ratio, random_seed, testset):
    SEED = random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True # ?
    torch.backends.cudnn.benchmark = False # ?
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(SEED))
    print('Ratio of datasets {} : {} : {}'.format(len(train_set), len(val_set), len(testset)))

    return train_set, val_set, testset

def preprocess(text):
    #text = re.sub('[-=+.#:^$*&\(\)\[\]\<\>]', '', text)
    text = re.sub('[.,–]', '', text)
    return text

def clean_sentence(df, clean_func):
    df['sentence'] = df['sentence'].apply(clean_func)
    return df


# for insert mode