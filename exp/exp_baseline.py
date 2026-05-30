# exp_stocknet.py
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from methods.stocknet import StockNet
from generator.generator import data_generator
from loss_func import CELoss, StocknetLoss, AlstmLoss, TransamLoss, DTMLLoss, ESPMPLoss
from globals import *
from methods.stocknet import StockNet
from methods.alstm import ALSTM
from methods.transam import TransAm
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from methods.indexgan import IndexGAN
import torch.optim as optim
from methods.dtml import DTML
from methods.taureau import Taureau
import torch.nn as nn
from methods.espmp import ESPMP 

def train_baseline(train_dataloader, bert, eps=0.5, epoches=10, method='stocknet', dataset_dir='dataset/', dataset_type='CNStock/', name='000001'):
    if method == 'stocknet':
        model = StockNet(bert).to(device)
        criterion = StocknetLoss()
    elif method == 'alstm':
        model = ALSTM(bert, use_attention=False, use_hinge_loss=False)
        criterion = AlstmLoss()
    elif method == 'atten_lstm':
        model = ALSTM(bert, use_attention=True)
        criterion = AlstmLoss()
    elif method == 'transam':
        model = TransAm(bert, feature_size=transam_config['vocab_size']+transam_config['price_size'])
        criterion = TransamLoss()
    elif method == 'dtml':
        # model = DTML(bert)
        model = ALSTM(bert, use_attention=True, weekday_info=True)
        criterion = DTMLLoss() 
    elif method == 'espmp':
        model = ESPMP(bert)
        criterion = ESPMPLoss()
        
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(_params, lr=lr, weight_decay=decay)
    model.train()
    train_loss, n_correct, n_train = 0, 0, 0
    

    for epoch in range(epoches):
        for news, price, targets in train_dataloader:
            news = {k: v.to(device) for k, v in news.items()}
            # news = cat_nested_sequence(news)
            # news = torch.stack([torch.stack(i) for i in news])
            price = torch.stack(price).to(device)
            targets = targets.to(device)
            if method =='espmp':
                outputs = model(news, price, targets)
            else:
                outputs = model(news, price)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            if method == 'stocknet':
                outputs = outputs[0]
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()  # stocknet为outputs[0]
            n_train += targets.size(0)
        print('Epoch [{}/{}], Loss: {}'.format(epoch+1, epoches, train_loss / n_train))

    # torch.save(model, f'models/{dataset_type}{name}_method_{method}_eps_{eps}_epoches_{epoches}_model.pkl')
    return model


def test_baseline(train_df, test_df, train_dataloader, test_dataloader, bert, \
    action_nextday = False, retrain = True, eps=0.5, epoches=20, method='dualcl',\
        dataset_dir=dataset_dir, dataset_type=dataset_type, name='000001'):
    
    if os.path.exists(f'models/{dataset_type}{name}_method_{method}_eps_{eps}_epoches_{epoches}_model.pkl') and retrain == False:
        model = torch.load(f'models/{dataset_type}{name}_method_{method}_eps_{eps}_epoches_{epoches}_model.pkl')
    else:
        if method in ['taureau']:
            model = Taureau(dataset_type)
        else:
            model = train_baseline(train_dataloader, bert, eps=eps, epoches=epoches, method=method, \
            dataset_dir=dataset_dir, dataset_type=dataset_type, name=name)
    
    pred_action = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for news, price, targets in test_dataloader:
            if method in ['taureau']:
                news = news
            else:
                news = {k: v.to(device) for k, v in news.items()}
            # news = cat_nested_sequence(news)
            # news = torch.stack([torch.stack(i) for i in news])
            price = torch.stack(price).to(device)
            targets = targets.to(device)
            outputs = model(news, price)
            targets_list.append(targets.cpu().numpy().flatten())
            if method == 'stocknet':
                predicted_labels = torch.argmax(outputs[0], dim=-1)
                pred_action.append(predicted_labels.cpu().numpy().flatten())
            elif method in ['alstm', 'atten_lstm', 'dtml', 'transam']:
                predicted_labels = torch.argmax(outputs, dim=-1)
                pred_action.append(predicted_labels.cpu().numpy().flatten())
            # elif method in ['transam']:
            #     pred_action.append(outputs.cpu().flatten())
            elif method == 'taureau':
                pred_action.append(outputs)

    action_sequence = np.concatenate(pred_action)
    targets_list = np.concatenate(targets_list)
    
    if action_nextday: 
        action_sequence = action_sequence.shift(1).fillna(0).astype("int")
    
    # Long-flat protocol: bullish (1) -> long (1), bearish (0) -> flat (0)
    action_sequence = [1 if x == 1 else 0 for x in action_sequence]
    targets_list = [1 if x == 1 else 0 for x in targets_list]
    
    test_df = test_df.iloc[LookBack-1:,:]
    test_df['action'] = action_sequence # predict action: 1=long, 0=flat
    test_df['labels'] = targets_list
                
    return test_df

    
def cat_nested_sequence(tokens, dim=1, target_length=256):
    if isinstance(tokens, list):
        if all(isinstance(token, list) for token in tokens):
            return [cat_nested_sequence(token, dim) for token in tokens]
        else:
            sequence = torch.cat(tokens, dim)[0]
            if len(sequence) < target_length:
                sequence = torch.cat([sequence, torch.zeros(target_length - len(sequence), *sequence[0].shape)], dim=0)
            elif len(sequence) > target_length:
                sequence = sequence[:target_length]
    else:
        sequence = tokens

    return sequence.to(device)