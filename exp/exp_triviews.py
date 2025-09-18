import torch
from tqdm import tqdm
from methods.triviews import TriViews_Model
import pickle
from loss_func import CELoss, SupConLoss, DualLoss, MSELoss, InfoNCELoss, TemporalSupConLoss, TemporalDualLoss
import pandas as pd
from transformers import BertModel, BertTokenizer, AutoModel
from generator.generator import data_generator
import os
import numpy as np
import torch.nn as nn
from torch import Tensor
from globals import *
import matplotlib.pyplot as plt
import seaborn as sns
from methods.bge import BGEM3
from methods.bert import BERT_text


def train_triviews(train_dataloader, eps=0.5, epoches=10, method='b4', dataset_dir='dataset/', 
                   dataset_type='CNStock/', name='000001', lookback=LookBack, lookahead=LookAhead,
                   freeze_ornot=True, series_type='price', text_type='news', alpha=0.5, temp=0.1, losspull=0):
    # basemodel_path = f'models/{dataset_type}/basemodel_{method}_lookahead{lookahead}_lookback{lookback}.pkl'
    
    if series_type == 'price': d_model =4
    else: d_model = 3

    if dataset_type=='CNStock':base_model = BertModel.from_pretrained('bert-chinese')
    else: base_model = BertModel.from_pretrained('bert-uncased')
    # print(f'The base model is {basemodel_path}')
    if method == 'bert':
        model = BERT_text(base_model, num_classes).to(device)
        criterion = DualLoss(alpha, temp)
    elif method == 'bert2':
        model = BERT_text(base_model, num_classes).to(device)
        criterion = CELoss()
    elif method == 'bert3':
        model = BERT_text(base_model, num_classes).to(device)
        criterion = SupConLoss(alpha, temp)
    else:
        model = TriViews_Model(base_model, num_classes, method, d_model, freeze_ornot=freeze_ornot).to(device)
            
        if method == 'ce':
            criterion = CELoss()
        elif method == 'scl':
            criterion = SupConLoss(alpha=alpha, temp=temp, losspull=losspull)
        elif method in ['b4']:
            criterion = DualLoss(alpha=alpha, temp=temp, losspull=losspull)
        elif method == 'mse':
            criterion = MSELoss()
        elif method == 'infonce':
            criterion = InfoNCELoss()
        else:
            raise ValueError('unknown method')
    
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(_params, lr=lr, weight_decay=decay)
    train_loss, n_correct, n_train = 0, 0, 0
    
    
    model.train()
    for epoch in range(epoches):
        for news, price, targets in train_dataloader:
            news = {k: v.to(device) for k, v in news.items()}
            price = torch.stack(price).to(device)
            targets = targets.to(device)

            if method in ['b4','scl']:
                outputs, base_model = model(news, price)
            else:
                outputs = model(news, price)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)
        print('Epoch [{}/{}], Loss: {}'.format(epoch+1, epoches, train_loss / n_train))

    # if freeze_ornot == True:
    #     torch.save(base_model, f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_losspull{losspull}_epoches{epoches}.pkl')
    # torch.save(model, f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_losspull{losspull}_epoches{epoches}.pkl')
    # torch.save(model, f'model/{name}.pkl')
    return model


def test_triviews(train_df, test_df, train_dataloader, test_dataloader, \
    action_nextday = False, retrain = True, eps=0.5, epoches=20, method='b4',\
        dataset_dir=dataset_dir, dataset_type=dataset_type, name='000001',
        lookahead=LookAhead, lookback=LookBack, freeze_ornot=True, series_type='price', text_type='news', alpha=0.5, temp=0.1, losspull=0):
    
    if os.path.exists(f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_losspull{losspull}_epoches{epoches}.pkl') and retrain == False:
        model = torch.load(f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_losspull{losspull}_epoches{epoches}.pkl')
    else:
        model = train_triviews(train_dataloader, eps=eps, epoches=epoches, method=method, \
            dataset_dir=dataset_dir, dataset_type=dataset_type, name=name,
            lookahead=lookahead, lookback=lookback, freeze_ornot=freeze_ornot, series_type=series_type, text_type=text_type, \
                alpha=0.5, temp=0.1, losspull=losspull)
    
    pred_action = []
    bull_views = []
    bear_views = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for news, price, targets in test_dataloader:
            # news = cat_nested_sequence(news)
            news = {k: v.to(device) for k, v in news.items()}
            price = torch.stack(price).to(device)
            targets = targets.to(device)
            if method in ['b4', 'scl']:
                outputs, base_model = model(news, price) 
            else:
                outputs = model(news, price)
            pred_action.append(torch.argmax(outputs['predicts'], -1).cpu().numpy().flatten())   
            targets_list.append(targets.cpu().numpy().flatten())

            if method in ['b4']:
                bull_views.append(outputs['bull_views'].cpu().numpy())
                bear_views.append(outputs['bear_views'].cpu().numpy())
    
    action_sequence = np.concatenate(pred_action)
    targets_list = np.concatenate(targets_list)
    
    if action_nextday: 
        action_sequence = action_sequence.shift(1).fillna(0).astype("int")
    
    action_sequence =  [-1 if x == 0 else x for x in action_sequence]
    targets_list = [-1 if x == 0 else x for x in targets_list]
    
    test_df = test_df.iloc[LookBack-1:,:]
    test_df['action'] = action_sequence # predict action
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