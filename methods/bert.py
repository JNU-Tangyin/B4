import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from globals import device

class BERT_text(nn.Module):
    def __init__(self, base_model, num_classes):
        super(BERT_text, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes).to(device)

    def forward(self, news, price):
        hiddens = self.base_model(**news, return_dict=True).last_hidden_state
        cls_feats = hiddens[:, 0, :]  # Using [CLS] token for classification
        label_feats = hiddens[:, 1:self.num_classes+1, :]
        predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        outputs = {
            'predicts': predicts,  
            'cls_feats': cls_feats,  
            'label_feats': label_feats 
        }

        return outputs
