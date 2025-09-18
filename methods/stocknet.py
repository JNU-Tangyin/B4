# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from globals import stocknet_config, device, train_batch_size

# class StockNet(nn.Module):
#     def __init__(self, bert):
#         super(StockNet, self).__init__()
#         self.config = stocknet_config
#         self.bert = bert
        
#         # Embedding layer
#         self.word_embed = nn.Embedding(stocknet_config['vocab_size'], stocknet_config['word_embed_size'])

#         # Define other layers
#         self.mel_h_size = stocknet_config['mel_h_size']
#         self.msg_embed_size = stocknet_config['msg_embed_size']
#         self.g_size = stocknet_config['g_size']
#         self.y_size = stocknet_config['output_size']

#         # Message Embedding Layer
#         self.message_embedding_layer = nn.GRU(stocknet_config['word_embed_size'], self.msg_embed_size, bidirectional=True)

#         # VMD (Variational Movement Decoder)
#         self.vmd_cell = nn.GRU(self.msg_embed_size * 2, self.mel_h_size)  # Assuming bidirectional
#         self.fc_z = nn.Linear(self.mel_h_size, stocknet_config['z_size'] * 2)  # Mean and log variance

#         # Output layer
#         self.fc_out = nn.Linear(self.mel_h_size + stocknet_config['price_embed_size'], self.y_size)

#         self.init_weights()

#     def init_weights(self):
#         # Weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Embedding):
#                 nn.init.xavier_uniform_(m.weight.data)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0)

#     def forward(self, news, price):
#         # Embedding lookup
#         embedded = self.bert.get_input_embeddings()(news['input_ids'].to(device))
#         # embedded = self.word_embed(embedded)

#         # Message Embedding Layer
#         # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, len(news), batch_first=True)
#         # msg_embeddings, _ = self.message_embedding_layer(packed_embedded)
#         # msg_embeddings, _ = nn.utils.rnn.pad_packed_sequence(msg_embeddings, batch_first=True)
#         msg_embeddings, _ = self.message_embedding_layer(embedded)
        
#         # VMD
#         vmd_input = msg_embeddings
#         vmd_output, _ = self.vmd_cell(vmd_input)
#         z_params = self.fc_z(vmd_output)

#         # Output layer
#         out = self.fc_out(vmd_output)
#         return out #, z_params


class StockNet(nn.Module):
    def __init__(self, bert):
        super(StockNet, self).__init__()
        self.config = stocknet_config
        self.bert = bert.to(device)
        self.lstm = nn.LSTM(stocknet_config['vocab_size'], stocknet_config['msg_embed_size'])
        # Embedding layer
        self.word_embed = nn.Embedding(stocknet_config['vocab_size'], stocknet_config['word_embed_size'])

        # Define other layers
        self.mel_h_size = stocknet_config['mel_h_size']
        self.msg_embed_size = stocknet_config['msg_embed_size']
        self.g_size = stocknet_config['g_size']
        self.y_size = stocknet_config['output_size']

        # Message Embedding Layer
        self.message_embedding_layer = nn.GRU(stocknet_config['word_embed_size'], self.msg_embed_size, bidirectional=True, batch_first=True)

        # VMD (Variational Movement Decoder)
        self.vmd_cell = nn.GRU(self.msg_embed_size * 2, self.mel_h_size, batch_first=True)  # Assuming bidirectional
        self.fc_z = nn.Linear(self.mel_h_size, stocknet_config['z_size'] * 2)  # Mean and log variance

        # Output layer
        self.fc_out = nn.Linear(self.mel_h_size + stocknet_config['price_embed_size'], self.y_size)
        self.fc_reduce = nn.Linear(256, 1)
        self.init_weights()

    def init_weights(self):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, news, price):
        # Embedding lookup
        embedded = self.bert.get_input_embeddings()(news['input_ids'].to(device))

        # Message Embedding Layer
        msg_embeddings, _ = self.message_embedding_layer(embedded)
        
        # VMD
        vmd_output, _ = self.vmd_cell(msg_embeddings)
        z_params = self.fc_z(vmd_output)
        
        # Extract mean and log variance for reparameterization
        mu, log_var = torch.chunk(z_params, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick

        # Price embedding
        price = price.to(device)
        # price_embed = self.fc_out(price)
        
        if vmd_output.size(1) < 256:
            padding = 256 - vmd_output.size(1)
            vmd_output = torch.nn.functional.pad(vmd_output, (0, 0, 0, padding), 'constant', 0)

        price_expanded = torch.zeros(price.shape[0], 256, 3).to(device)
        price_expanded[:, :20, :] = price 
        combined_input = torch.cat((vmd_output, price_expanded), dim=-1)
        reduced_combined_input = self.fc_reduce(combined_input.transpose(1, 2)).squeeze(-1)

        # Output layer
        # out = self.fc_out(combined_input)
        out = self.fc_out(reduced_combined_input)
        predicted_labels = torch.argmax(out, dim=-1)

        # Return outputs and latent parameters
        return out, mu, log_var

