# Aspect-Based Sentiment Analysis

### Task
**ABSA**  
ASC: Aspect-level Sentiment Analysis  
In this repo., Aspect means Aspect Category or Term

### Dataset
- SemEval-16 restaurant
- SemEval-16 laptop
- sentihood

### Models
**bert_intermediate.py**  
1. Bert_Base: basic BertForSequenceClassification
2. Bert_Base_rpos: only for Term Classification. Relative Position id: \[200,4,3,2,1,0,0,1,2,3,4,200,200,200]  200: not sentence words including special tokens. max_seq_length value. 0 means aspect terms in sentence.
3. Bert_Attention: Self Att layer for Bert intermediate hidden states (paper title)

**bert_attscores.py**  
Additional layer to use high attention score words. Attention scores from mean of all attention heads in final self-attention layer, and use final hidden states of that words.  
1. Bert_AttScore: mean of top-k attention score words to classifier. 
2. Bert_AttScore_RNN: use bi-GRU for top-k attention score words sequential representation.
3. Bert_AttScore_RNN_add: top-k attention score words and some special tokens to bi-GRU. <SEP>, <CLS>, <ASP words> and so on.
