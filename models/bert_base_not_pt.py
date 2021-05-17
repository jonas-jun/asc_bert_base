import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

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




# for inser mode