import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification

class Bert_Base(nn.Module):
    def __init__(self, num_classes):
        super(Bert_Base, self).__init__()
        self.num_classes = num_classes
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
            output_hidden_states=False,
            output_attentions=False,
            num_labels=self.num_classes)
        print('Bert Model Loaded')

    def forward(self, input_ids, att_mask, token_ids, labels):
        loss, out = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
        labels=labels, return_dict=False)
        return loss, out

class Bert_Base_rpos(nn.Module):
    def __init__(self, num_classes):
        super(Bert_Base_rpos, self).__init__()
        self.num_classes = num_classes
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
            output_hidden_states=False,
            output_attentions=False,
            num_labels=self.num_classes)
        print('Bert Model Loaded')

    def forward(self, input_ids, att_mask, token_ids, pos_ids, labels):
        loss, out = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_ids,
            position_ids=pos_ids, labels=labels, return_dict=False)
        return loss, out


class Bert_Attention(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hid_dim=256):
        super(Bert_Attention, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.num_layers = opt.num_layers
        self.fc_hid_dim = fc_hid_dim
        self.dropout = nn.Dropout(0.1)
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                             output_hidden_states=True,
                                             output_attentions=False)
        self.device = opt.device
        print('BERT Model Loaded')
        
        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embed_dim))
        #print('q_t shape: ', q_t.shape)
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        #print('self.q shape: ', self.q.shape)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embed_dim, self.fc_hid_dim))
        #print('w_ht shape: ', w_ht.shape)
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)
        #print('self.w_h shape', self.w_h.shape)
        
        self.fc = nn.Linear(self.fc_hid_dim, self.num_classes)
        
    def forward(self, input_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, hidden_states = self.bert(input_ids=input_ids,
                                                                    attention_mask=att_mask,
                                                                    token_type_ids=token_ids,
                                                                    return_dict=False)
        #print('last_hidden_state shape: ', last_hidden_state.shape) # [8, 80, 768]
        #print('pooler_output shape: ', pooler_output.shape) # [8, 768]
        #print('hidden_states shape: ', hidden_states.shape) # tuple
        
        # num_layer 만큼의 hidden_states들을 stack해서 attention 적용

        hidden_states = torch.stack([hidden_states[-layer_i][:,0].squeeze()\
                                    for layer_i in range(1, self.num_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_layers, self.embed_dim)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        #print('v shape: ', v.shape) # [8, 12]
        v = F.softmax(v, -1)
        #print('v Softmaxed: ', v.shape) # [8, 12]
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        #print('v_temp shape: ', v_temp.shape) # [8, 768, 1]
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        #print('final v shape: ', v.shape) [8, 256]
        return v


# for insert mode