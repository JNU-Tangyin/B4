import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from sklearn.utils import shuffle
from globals import alstm_config, device, train_batch_size, dtml_config

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.hidden_size = alstm_config['hidden_size']
        self.attn_fc = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.attn_combine_fc = nn.Linear(self.hidden_size , 1).to(device)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn_fc(lstm_output))
        attn_weights = self.attn_combine_fc(attn_weights).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        new_hidden = torch.sum(attn_weights.unsqueeze(2) * lstm_output, dim=1)
        return new_hidden
    
class ALSTM(nn.Module):
    def __init__(self, bert, use_attention=False, use_hinge_loss=False, weekday_info=False):
        super(ALSTM, self).__init__()
        self.word_embed = nn.Embedding(alstm_config['vocab_size'], alstm_config['word_embed_size']).to(device)
        self.use_attention = use_attention
        self.use_hinge_loss = use_hinge_loss
        self.weekday_info = weekday_info
        # self.input_size = alstm_config['input_size']
        self.hidden_size = alstm_config['hidden_size']
        self.output_size = alstm_config['output_size']
        self.lstm = nn.LSTM(alstm_config['word_embed_size'], self.hidden_size, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(device)
        self.bert = bert.to(device)
        
        if use_attention:
            # self.attention_fc = nn.Linear(self.hidden_size, self.hidden_size).to(device)
            # self.attention_combine = nn.Linear(self.hidden_size * 2, 1).to(device)
            self.attention = Attention()
            

    def forward(self, news, price):
        embeded = self.bert.get_input_embeddings()(news['input_ids'])
        # embeded = embeded.mean(dim=-1)
        if self.weekday_info:
            price_expanded = torch.zeros(price.shape[0], dtml_config['price_size'], dtml_config['word_embed_size']).to(device)
        else:
            price_expanded = torch.zeros(price.shape[0], alstm_config['price_size'], alstm_config['word_embed_size']).to(device)
        price_expanded[:, :, :20] = price.permute(0,2,1) 
        combined_embeded = torch.cat((embeded, price_expanded), dim=1)
        
        # price_expanded = price.reshape(price.shape[0], price.shape[1]*price.shape[2])
        # combined_embeded = torch.cat((embeded, price_expanded), dim=1)
        
        lstm_out, _ = self.lstm(combined_embeded)
        if self.use_attention:
            lstm_out = self.attention(lstm_out)
        else:
            lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        if self.use_hinge_loss:
            output = torch.clamp(output, min=0)
        return output