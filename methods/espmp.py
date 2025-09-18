import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch
import torch.nn.functional as F

class ESPMP(nn.Module):
    def __init__(self, base_model, num_classes=2, lstm_hidden_size=30, attention_heads=1):
        super(ESPMP, self).__init__()
        self.base_model = base_model 
        self.num_classes = num_classes


        self.text_lstm = nn.LSTM(input_size=base_model.config.hidden_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.price_lstm = nn.LSTM(input_size=3, hidden_size=lstm_hidden_size, batch_first=True)  # 假设价格有3个指标

        self.text_attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=attention_heads)
        self.price_attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=attention_heads)

        self.linear = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, text, price, targets):
        text_outputs = self.base_model(**text)
        text_hiddens = text_outputs.last_hidden_state
        cls_feats = text_hiddens[:, 0, :]  

        text_lstm_out, _ = self.text_lstm(cls_feats.unsqueeze(1))
        text_attention_out, _ = self.text_attention(text_lstm_out, text_lstm_out, text_lstm_out)

        price_lstm_out, _ = self.price_lstm(price)
        price_attention_out, _ = self.price_attention(price_lstm_out, price_lstm_out, price_lstm_out)

        # combined_feats = torch.cat([text_attention_out[:, 0, :], price_attention_out[:, 0, :]], dim=-1)
        # label_feats = combined_feats[:, 1:self.num_classes+1, :]
        # predicts = torch.einsum('bd,bcd->bc', combined_feats, label_feats)
        
        combined_feats = text_attention_out[:, 0, :]
        label_feats = text_attention_out[:, 1:self.num_classes+1, :]
        predicts = torch.einsum('bd,bcd->bc', text_attention_out[:, 0, :], label_feats)
        anchor, positive, negative = triplet_selector(price, targets, l=5)


        outputs = {
            'predicts': predicts,
            'cls_feats': combined_feats,
            'label_feats': label_feats,
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }
        return outputs



def euclidean_distance(x1, x2):
    return torch.sqrt(((x1 - x2) ** 2).sum(dim=1))

def pearson_corr(x1, x2):
    x1_mean = torch.mean(x1, dim=1, keepdim=True)
    x2_mean = torch.mean(x2, dim=1, keepdim=True)
    cov = torch.sum((x1 - x1_mean) * (x2 - x2_mean), dim=1)
    std_x1 = torch.sqrt(torch.sum((x1 - x1_mean) ** 2, dim=1))
    std_x2 = torch.sqrt(torch.sum((x2 - x2_mean) ** 2, dim=1))
    return cov / (std_x1 * std_x2 + 1e-8)

# 制作 anchor, positive, negative 样本
def triplet_selector(prices, labels, l):
    batch_size = prices.size(0)
    anchor_indices = list(range(batch_size))

    # 计算 pairwise Pearson 相关性
    pairwise_corr = torch.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            pairwise_corr[i, j] = pearson_corr(prices[i][-l:], prices[j][-l:])

    # 计算欧氏距离
    pairwise_dist = torch.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            pairwise_dist[i, j] = euclidean_distance(prices[i][-l:], prices[j][-l:])

    anchor, positive, negative = [], [], []

    for i in anchor_indices:
        # 获取与 anchor 相关性最高的 k 个样本（价格收益相似）
        same_label_indices = (labels == labels[i]).nonzero().squeeze()
        opposite_label_indices = (labels != labels[i]).nonzero().squeeze()

        # Positive: 选择方向一致且距离最小的样本
        same_label_distances = pairwise_dist[i, same_label_indices]
        pos_index = same_label_indices[torch.argmin(same_label_distances)]
        positive.append(prices[pos_index])

        # Negative: 选择方向相反且距离最小的样本
        opposite_label_distances = pairwise_dist[i, opposite_label_indices]
        neg_index = opposite_label_indices[torch.argmin(opposite_label_distances)]
        negative.append(prices[neg_index])

        anchor.append(prices[i])

    return torch.stack(anchor), torch.stack(positive), torch.stack(negative)


prices = torch.randn((16, 10, 5))  # 假设 16 个样本, 每个样本有 10 天的价格数据, 5 个特征
labels = torch.randint(0, 2, (16,))  # 二分类标签，0 和 1
l = 5  # 使用过去 5 天的趋势

