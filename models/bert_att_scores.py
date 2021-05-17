import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Bert_Att_only(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hid_dim=128, top_k=3, att_pooling='mean'):
        super(Bert_Att_only, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.top_k = top_k
        self.att_pooling = att_pooling
        if att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        #self.dropout = nn.Dropout(p=0.5)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids, pos_ids, last_ids):
        '''
        args
        pos_ids: [3,2,1,0,0,1,2,3,4,5,maxlength, maxlength, ...] 0 means aspect words
        last_ids: last <SEP> token id, it means lengths
        '''

        asp_ids = list()
        for i in pos_ids:
            ids = (pos_ids==0).nonzero(as_tuple=True)[0].tolist()
            asp_ids.append(ids) # aspect word ids, pos_ids==0
        
        output_dict = self.bert(input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        atts = output_dict.attentions[-1]
        in_batch_atts = list()
        for a in atts:
            in_batch_atts.append(torch.mean(a, dim=0)) # average of all att heads, each (batch_size, max_length, max_length)
        top_k_idx = list()
        for att, asp, last in zip(in_batch_atts, asp_ids, last_ids):
            sum_ = sum(att[asp[0]:(asp[-1]+1), :]) # sum attention scores for multi-aspect words (1,200)
            idxs = torch.sort(sum_[1:last+1], descending=True).indices[:self.top_k] # exclude 0:<CLS>, last:<SEP>, +len(asp)
            top_k_idx.append(idxs+1) # re consider 0:<CLS>
        # len(top_k_idx): batch_size
        # top_k_idx[0].shape: [1,3]

        # get top-k hidden states
        hids = output_dict.last_hidden_state
        output = get_hiddens(last_hiddens=hids, idx_list=top_k_idx, pooling=self.att_pooling) # only top-k att score words
        output = self.fc1(output)
        return output


class Bert_Att_Aspwords(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hidden_dim=128, top_k=3, att_pooling='mean'):
        super(Bert_Att_only, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.top_k = top_k
        self.att_pooling = att_pooling
        if att_pooling == 'concat':
            self.fc1 = nn.Linear((self.embed_dim) * top_k, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        #self.dropout = nn.Dropout(p=0.5)
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids, pos_ids, last_ids):
        '''
        args
        pos_ids: [3,2,1,0,0,1,2,3,4,5,maxlength, maxlength, ...] 0 means aspect words
        last_ids: last <SEP> token id, it means lengths
        '''

        asp_ids = list()
        for i in pos_ids:
            ids = (pos_ids==0).nonzero(as_tuple=True)[0].tolist()
            asp_ids.append(ids) # aspect word ids, pos_ids==0
        
        output_dict = self.bert(input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            output_attentions=True, encoder_hidden_states=True, return_dict=True)

        # get top-k att idx in final att layer
        atts = output_dict.attentions[-1]
        in_batch_atts = list()
        for a in atts:
            in_batch_atts.append(torch.mean(a, dim=0)) # average of all att heads, each (batch_size, max_length, max_length)
        top_k_plus_aspwords = list()
        for att, asp, last in zip(in_batch_atts, asp_ids, last_ids):
            sum_ = sum(att[asp[0]:(asp[-1]+1), :]) # sum attention scores for multi-aspect words (1,200)
            idxs = torch.sort(sum_[1:last+1], descending=True).indices[:self.top_k] # exclude 0:<CLS>, last:<SEP>
            plus_aspwords = list(set(asp + idxs+1)) # re consider 0:<CLS>, # delete duplicate (asp words)
            top_k_plus_aspwords.append(plus_aspwords) 
        # len(top_k_idx): batch_size
        # top_k_idx[0].shape: [1,3]
        

        # get top-k hidden states
        hids = output_dict.last_hidden_state
        output = get_hiddens(last_hiddens=hids, idx_list=top_k_idx, pooling=self.att_pooling) # only top-k att score words
        output = self.fc1(output)
        return output

# top-k개가 아니라 asp_words의 길이 + k개?



def get_hiddens(last_hiddens, idx_list, pooling='mean'):
    '''
    @args
    last_hiddens: bert last hidden states
    idx_list: idxs to get hids, top-k words or top-k words + aspect words
    pooling: how get final rep. vetors, 'mean', 'sum', 'concat'
    '''
    final = list()
    for idx, hid in zip(idx_list, last_hiddens):
        if pooling=='sum':
            final.append(torch.sum(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='mean' or pooling=='average':
            final.append(torch.mean(hid[idx, :], dim=0).unsqueeze(0)) # (1, 768)
        elif pooling=='concat':
            final.append(hid[idx, :].view(1, -1)) # (1, 768*k)
    final = torch.cat(final, dim=0) # to tensor (batch_size, 768) or (batch_size, 768*k) (concat)
    return final





# for insert mode