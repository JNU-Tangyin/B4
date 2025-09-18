import torch
import torch.nn as nn
from torch import Tensor
from math import sqrt
import math
from globals import *
import torch.nn.functional as F


class B4(nn.Module):

    def __init__(self, base_model, num_classes, method, d_model, freeze_ornot = True):
        super().__init__()
        self.base_model = base_model.to(device)
        self.num_classes = num_classes
        self.method = method
        # self.lstm = nn.LSTM(base_model.config.hidden_size, base_model.config.hidden_size, batch_first=True).to(device)
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(freeze_ornot)
        
        self.word_embeddings = self.base_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, d_ff, d_llm)

    def forward(self, news, price):
        if self.method == 'price_dualcl':
            news_embeddings = self.base_model.get_input_embeddings()(news['input_ids'][:,:3].to(device)) 
        else:    
            news_embeddings = self.base_model.get_input_embeddings()(news['input_ids'].to(device))
            
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(price, source_embeddings, source_embeddings)
        bert_enc_out = torch.cat([news_embeddings, enc_out], dim=1)
        # lstm_out, _ = self.lstm(bert_enc_out)
        
        hiddens = self.base_model(inputs_embeds=bert_enc_out).last_hidden_state
        # hiddens = self.base_model(inputs_embeds=bert_enc_out).last_hidden_state
        
        cls_feats = hiddens[:, 0, :]
        
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
            outputs = {
                'predicts': predicts,
                'cls_feats': cls_feats,
                'feats': bert_enc_out
            }
            
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
            
            bull_views = torch.einsum('bcd, bd->bc', hiddens, label_feats[:,0,:]) / math.sqrt(label_feats[:,0,:].size(1))
            bull_views = torch.softmax(bull_views, dim=-1)
            bear_views = torch.einsum('bcd, bd->bc', hiddens, label_feats[:,1,:]) / math.sqrt(label_feats[:,1,:].size(1))
            bear_views = torch.softmax(bear_views, dim=-1)

            outputs = {
                'predicts': predicts,
                'cls_feats': cls_feats,
                'label_feats': label_feats,
                'bull_views': bull_views,
                'bear_views': bear_views,
                'feats': bert_enc_out
            }
        return outputs, self.base_model
    
    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


