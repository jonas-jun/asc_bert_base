import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification

class Bert_Position_after(nn.Module):
    def __init__(self, opt, embed_dim=768, fc_hidden_dim=256):
        super(Bert_Position_after, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pos_embedding = nn.Embedding(opt.max_length, self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        # self.fc2 = nn.Linear(self.fc_hidden_dim, self.num_classes)
        # self.relu = nn.ReLU()
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids, pos_ids, last_ids):
        output_dict = self.bert(input_ids, attention_mask=att_mask, token_type_ids=token_ids,
                encoder_hidden_states=True, return_dict=True)
        pos_embedded = self.pos_embedding(pos_ids) # (8, 200, 768)
        pos_added = output_dict.last_hidden_state + pos_embedded # (8, 200, 768)
        sum = torch.sum(pos_added, dim=1) # (8, 768)
        output = self.fc(sum)
        # output = self.relu(self.fc1(sum)) # (8, 256) fc_hidden_dim
        # output = self.fc2(output) # (8, 3) num_classes
        return output

class Bert_Relative_Pos(nn.Module):
    def __init__(self, opt, embed_dim=768): #fc_hidden_dim=256):
        super(Bert_Relative_Pos, self).__init__()
        self.num_classes = opt.num_classes
        self.embed_dim = embed_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(self.embed_dim, self.num_classes)
        #self.fc2 = nn.Linear(self.fc_hidden_dim, self.num_classes)
        #self.relu = nn.ReLU()
        self.device = opt.device

    def forward(self, input_ids, att_mask, token_ids, pos_ids, last_ids):
        output_dict = self.bert(input_ids, attention_mask=att_mask, token_type_ids=token_ids,
                position_ids=pos_ids, encoder_hidden_states=True, return_dict=True)
        cls = output_dict.last_hidden_state[:, 0, :] # (8, 768)
        output = self.fc1(cls)
        #output = self.relu(self.fc1(cls)) # (8, 256) fc_hidden_dim
        #output = self.fc12(output) # (8, 3) num_classes
        return output

# for insert mode