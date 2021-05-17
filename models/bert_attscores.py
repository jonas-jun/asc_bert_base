import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Bert_AttScore(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hid_dim=128, top_k=3, att_head='all', att_pooling='mean'):
        assert type(att_head)==list or att_head=='all', "att_head should be 'all' or list type [3, 5]"
        super(Bert_AttScore, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hid_dim = fc_hid_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.top_k = top_k
        self.att_head = att_head
        self.att_pooling = att_pooling
        if self.att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids):
        '''

        '''
        # <SEP> ids
        sep_ids = [(i==102).nonzero(as_tuple=True)[0] for i in input_ids]
        # aspect words ids
        asp_ids = [list(range(s[0]+1, s[-1])) for s in sep_ids]
        
        output_dict = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        last_att = output_dict.attentions[-1]
        target_heads = last_att if self.att_head == 'all' else last_att[:, self.att_head, :, :]
        atts = [torch.mean(i, dim=0) for i in target_heads] # len(atts): batch_size, atts[0].shape: (200, 200)
        
        # get top-k score words ids
        top_k_words = list() # len 16 list, [0]: len top-k list [14, 3, 17]
        for att, asp, sep in zip(atts, asp_ids, sep_ids):
            att_score = sum(att[asp, :]) # (200), sum att scores of asp words
            top_k_idx = torch.sort(att_score[1:sep[0]], descending=True).indices[:self.top_k] # (3), exclude [CLS], [SEP], aspect words
            top_k_words.append(top_k_idx+1) # consider [CLS] token for idx
        
        # get top-k hidden states
        hids = output_dict.last_hidden_state
        output = get_hiddens(last_hids=hids, top_k_list=top_k_words, pooling=self.att_pooling) # only top-k att score words
        output = self.fc1(output)
        return output, top_k_words

class Bert_AttScore_RNN(nn.Module):
    def __init__(self, opt, embed_dim=768, rnn_hid_dim=256, fc_hid_dim=128, bidirectional=True, top_k=3,
            att_head='all', att_pooling='gru'):
        super(Bert_AttScore_RNN, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.rnn_hid_dim = rnn_hid_dim
        self.fc_hid_dim = fc_hid_dim
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn_pool = True if att_pooling in ['rnn', 'lstm', 'gru'] else False
        if self.rnn_pool:
            self.rnn = nn.GRU(input_size=768, hidden_size=rnn_hid_dim, num_layers=1, bidirectional=self.bidirectional, 
                batch_first=True)
        self.top_k = top_k
        self.att_head = att_head
        self.att_pooling = att_pooling
        if self.att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        elif self.att_pooling in ['rnn', 'lstm', 'gru']:
            self.fc1 = nn.Linear(self.rnn_hid_dim*(self.bidirectional+1), self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids):
        # <SEP> ids
        sep_ids = [(i==102).nonzero(as_tuple=True)[0] for i in input_ids]
        # aspect words ids
        asp_ids = [list(range(s[0]+1, s[-1])) for s in sep_ids]
        
        output_dict = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        last_att = output_dict.attentions[-1]
        target_heads = last_att if self.att_head == 'all' else last_att[:, self.att_head, :, :]
        atts = [torch.mean(i, dim=0) for i in target_heads] # len(atts): batch_size, atts[0].shape: (200, 200)
        
        # get top-k score words ids
        top_k_words = list() # len 16 list, [0]: len top-k list [14, 3, 17]
        for att, asp, sep in zip(atts, asp_ids, sep_ids):
            att_score = sum(att[asp, :]) # (200), sum att scores of asp words
            top_k_idx = torch.sort(att_score[1:sep[0]], descending=True).indices[:self.top_k] # (3), exclude [CLS], [SEP], aspect words
            if self.rnn_pool:
                while len(top_k_idx) < self.top_k:
                    top_k_idx = top_k_idx.tolist()
                    top_k_idx += [top_k_idx[0]]
                    top_k_idx = torch.tensor(top_k_idx) # append first token if length of sentence is less than top_k
                top_k_idx = torch.sort(top_k_idx, descending=False).values # order
            top_k_words.append(top_k_idx+1) # consider [CLS] token for idx
        
        # get top-k hidden states
        hids = output_dict.last_hidden_state
        output = get_hiddens(last_hids=hids, top_k_list=top_k_words, pooling=self.att_pooling) # only top-k att score words
        
        # gru에 집어 넣는데.. (16, 3, 768)이 들어가서 (16, 3, 256)이 나온다. 여기서 [:, -1, :]만 사용할 것 (16, 256)
        output, _ = self.rnn(output)
        output = output[:, -1, :] # last output
        
        output = self.fc1(output)
        return output, top_k_words

class Bert_AttScore_RNN_add(nn.Module):
    def __init__(self, opt, embed_dim=768, rnn_hid_dim=256, fc_hid_dim=128, bidirectional=True, top_k=3,
            att_head='all', additional_token='asp', att_pooling='gru'):
        '''
        @Args
        bidirectional: rnn directions, [True, False]
        top_k: how many words to consider, default: 3
        additional_token = which token to add with high attention score tokens, ['asp', 'sep1', 'sep2', 'sep_both', 'cls']
        '''
        
        assert additional_token in ['asp', 'sep1', 'sep2', 'sep_both', 'cls']
        super(Bert_AttScore_RNN_add, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.rnn_hid_dim = rnn_hid_dim
        self.fc_hid_dim = fc_hid_dim
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rnn_pool = True if att_pooling in ['rnn', 'lstm', 'gru'] else False
        if self.rnn_pool:
            self.rnn = nn.GRU(input_size=768, hidden_size=rnn_hid_dim, num_layers=1, bidirectional=self.bidirectional, 
                batch_first=True) # dropout 제외 one layer
        self.top_k = top_k
        self.att_head = att_head
        self.additional_token = additional_token
        self.att_pooling = att_pooling
        if self.att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        elif self.att_pooling in ['rnn', 'lstm', 'gru']:
            self.fc1 = nn.Linear(self.rnn_hid_dim*(self.bidirectional+1), self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids):
        # <SEP> ids
        sep_ids = [(i==102).nonzero(as_tuple=True)[0] for i in input_ids]
        sep_ids_2 = [(i==102).nonzero(as_tuple=True)[-1] for i in input_ids]
        # aspect words ids
        asp_ids = [list(range(s[0]+1, s[-1])) for s in sep_ids]
        
        output_dict = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        last_att = output_dict.attentions[-1]
        target_heads = last_att if self.att_head == 'all' else last_att[:, self.att_head, :, :]
        atts = [torch.mean(i, dim=0) for i in target_heads] # len(atts): batch_size, atts[0].shape: (200, 200)
        
        # get top-k score words ids
        top_k_words = list() # len 16 list, [0]: len top-k list [14, 3, 17]
        for att, asp, sep in zip(atts, asp_ids, sep_ids):
            att_score = sum(att[asp, :]) # (200), sum att scores of asp words
            top_k_idx = torch.sort(att_score[1:sep[0]], descending=True).indices[:self.top_k] # (3), exclude [CLS], [SEP], aspect words
            if self.rnn_pool:
                while len(top_k_idx) < self.top_k:
                    top_k_idx = top_k_idx.tolist()
                    top_k_idx += [top_k_idx[0]]
                    top_k_idx = torch.tensor(top_k_idx) # append first token if length of sentence is less than top_k
                top_k_idx = torch.sort(top_k_idx, descending=False).values # order
            top_k_words.append(top_k_idx+1) # consider [CLS] token for idx
        
        # get top-k hidden states
        hids = output_dict.last_hidden_state
        if self.additional_token == 'asp':
            output = get_hiddens_asp(last_hids=hids, top_k_list=top_k_words, asp_ids=asp_ids, asp_pool='mean')
        elif self.additional_token == 'cls':
            output = get_hiddens_cls(last_hids=hids, top_k_list=top_k_words)
        elif self.additional_token in ['sep1', 'sep2', 'sep_both']:
            output = get_hiddens_sep(last_hids=hids, top_k_list=top_k_words, sep_ids=sep_ids, which=self.additional_token)
        
        # gru에 집어 넣는데.. (16, 3, 768)이 들어가서 (16, 3, 256)이 나온다. 여기서 [:, -1, :]만 사용할 것 (16, 256)
        output, _ = self.rnn(output)
        output = output[:, -1, :] # last output
        
        output = self.fc1(output)
        return output, top_k_words

def get_hiddens(last_hids, top_k_list, pooling='mean'):
    '''
    @args
    last_hids: bert last hidden states
    top_k_list: idxs to get hids, top-k words or top-k words + aspect words
    pooling: how get final rep. vetors, 'mean', 'sum', 'concat'
    '''
    final = list()
    for idx, hid in zip(top_k_list, last_hids):
        if pooling=='sum':
            final.append(torch.sum(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='mean' or pooling=='average':
            final.append(torch.mean(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='concat':
            final.append(hid[idx, :].view(1, -1)) # (1, 768*k)
        elif pooling in ['rnn', 'lstm', 'gru']:
            final.append(hid[idx, :].unsqueeze(0))
    final = torch.cat(final, dim=0) # to tensor (batch_size, 768) or (batch_size, 768*k) (concat)
    return final

def get_hiddens_cls(last_hids, top_k_list, pooling='rnn'):
    'CLS token + high scores'
    final = list()
    for idx, hid in zip(top_k_list, last_hids):
        idxs = [0] + idx.tolist()
        final.append(hid[idxs, :].unsqueeze(0))
    final = torch.cat(final, dim=0)
    return final

def get_hiddens_sep(last_hids, top_k_list, sep_ids, which='sep1', pooling='rnn'):
    'high scores + first SEP token'
    assert which in ['sep1', 'sep2', 'sep_both'], "which should be in ['sep1', 'sep2', 'sep_both']"
    final = list()
    for idx, sep, hid in zip(top_k_list, sep_ids, last_hids):
        if which == 'sep1':
            idxs = idx.tolist() + [sep[0].item()]
        if which == 'sep2':
            idxs = idx.tolist() + [sep[-1].item()]
        elif which == 'sep_both':
            idxs = idx.tolist() + [sep[0].item()] + [sep[-1].item()]
        final.append(hid[idxs, :].unsqueeze(0))
    final = torch.cat(final, dim=0)
    return final
        
def get_hiddens_asp(last_hids, top_k_list, asp_ids, asp_pool='mean', pooling='rnn'):
    'concat asp_ids mean & sum 바꾸기만 하면 됨!'
    final = list()
    for idx, asp, hid in zip(top_k_list, asp_ids, last_hids):
        highs = hid[idx, :] # (3, 768)
        asp_words = hid[asp, :] # (1~4, 768)
        asp_pooled = torch.mean(asp_words, dim=0).unsqueeze(0) if asp_pool=='mean' \
            else torch.sum(asp_words, dim=0).unsqueeze(0)
        total = torch.cat([highs, asp_pooled]).unsqueeze(0)
        final.append(total)
    final = torch.cat(final, dim=0)
    return final


# for insert mode