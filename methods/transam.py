import torch
import torch.nn as nn
import math
from globals import train_batch_size, transam_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) < 259:
            zeros = torch.zeros(x.size(0), 259 - x.size(1), dtype=x.dtype, device=x.device)
            x = torch.cat((x, zeros), dim=1)
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, bert, feature_size=30, num_layers=2, dropout=0.2):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)
        self.decoder = nn.Linear(feature_size, 2).to(device)
        self.linear = nn.Linear(train_batch_size, 1).to(device)
        self.init_weights()
        self.bert = bert.to(device)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, news, price):
        global device
        embeded = self.bert.get_input_embeddings()(news['input_ids'])#.to(device)
        price_expanded = torch.zeros(price.shape[0], transam_config['price_size'], transam_config['word_embed_size'])#.to(device)
        price_expanded[:, :, :20] = price.permute(0,2,1) 
        src = torch.cat((embeded.to(device), price_expanded.to(device)), dim=1)
        # src = torch.cat((news, price), dim=-1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src.mean(dim=-1))
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output[:,-1,:]
        # output = output.view(output.shape[0],-1)
        # output = self.linear(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask