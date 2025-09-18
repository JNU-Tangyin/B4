from transformers import BertModel, BertTokenizer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
from functools import partial
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
from load_data import *
from preprocess.price_process import *
from preprocess.text_process import *
# from sklearn.preprocessing import train_test_split
import itertools


def bge_embed(input, tokenizer):
    input = ''.join(input)
    output = tokenizer(input)
    return list(output)
    
def my_collate(batch, tokenizer, method, num_classes):
    tokens, price, label_ids = map(list, zip(*batch))
    # text_ids = tokenize_nested_sequence(tokens, tokenizer)
    if method in ['bge-m3']:
        text_ids = list(map(lambda x: bge_embed(x, tokenizer), tokens))
        text_ids = torch.tensor(text_ids)
    elif method in ['taureau']:
        text_ids = tokens
    else:
        text_ids = tokenizer(tokens, padding=True, truncation=True,
                         max_length=256, is_split_into_words=True,
                        #  is_split_into_words=False,
                         add_special_tokens=True, return_tensors='pt')
    # if method in ['dualcl']:
    #     positions = torch.zeros_like(text_ids['input_ids'])
    #     positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
    #     text_ids['position_ids'] = positions
    return text_ids, price, torch.tensor(label_ids)


def data_generator(dataset_dir, dataset_type ,name, train_size=0.7, train_batch_size=64, test_batch_size=64, 
                   method='dualcl', workers=0, lookahead=5, lookback=20, series_type='price', text_type='news'):
    price = load_price(dataset_dir=dataset_dir, dataset_type=dataset_type, name=name)
    news = load_news(dataset_dir=dataset_dir, dataset_type=dataset_type, name=name)    
    df = concat_events_and_price(price, news, selected_elements=['type','trigger','arguments'], \
                           concat_mode='inner', method=method, lookahead=lookahead, lookback=lookback,\
                           series_type=series_type, text_type=text_type)
    
    cut_line = round(len(df)*train_size)
    train_data = df.iloc[:cut_line]
    test_data = df.iloc[cut_line:]
    # train_data, test_data = train_test_split(df, test_size=1 - train_size, random_state=42)
    
        
    trainset = Dataset(train_data, method, lookback, series_type, text_type)
    testset = Dataset(test_data, method, lookback, series_type, text_type)

    if dataset_type =='USStock/': tokenizer = BertTokenizer.from_pretrained('bert-uncased')
    else: tokenizer = BertTokenizer.from_pretrained('bert-chinese')
                    
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=2)
    train_dataloader = DataLoader.DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader.DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    
    return train_data, test_data, train_dataloader, test_dataloader

class Dataset(Dataset):
    def __init__(self, raw_data, method, lookback, series_type, text_type):
        self.data = raw_data
        if series_type == 'price': self.price = raw_data[OCLH]
        elif series_type == 'indicators': self.price = raw_data[kdj]
        label_dict = {'down': 0, 'up': 1} 
        self.label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        self.sep_token = ['[SEP]'] 
        self.news = raw_data[['news']]
        self.label = raw_data[['label']]
        self.lookback = lookback

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.lookback
        
        news_data_list = self.news.iloc[start_index:end_index].astype(str).values.tolist()
        # news = [self.label_list + self.sep_token + item for item in news_data_list]
        # 如果是dualcl，一个窗口只需要标签增强一次，不需要每个样本都标签增强
        news = self.label_list+ self.sep_token +np.array(news_data_list)[0].astype(str).tolist()
        price = torch.tensor(self.price.iloc[start_index:end_index].values, dtype=torch.float32)
        label = torch.tensor(self.label.iloc[end_index-1], dtype=torch.long)  
        # date = self.data.reset_index().iloc[end_index-1]['date'].values
        return news, price, label

    def __len__(self):
        return len(self.data) - LookBack + 1
    
    def add_label_and_sep(self, row):
        return self.label_list + self.sep_token + [row.values[0]]    
    
################# concat price and event#############
def concat_events_and_price(price, text,  selected_elements=['type'], \
                           concat_mode='inner', method=False, lookahead=5, lookback=20,
                           series_type='price', text_type='news'):
   
    price = process_price(price, method=method, lookahead=lookahead)
    weekcol = ['weekday_'+str(i) for i in range(5)]
    price['label'] = price['reward'].apply(lambda x: 1 if x<0.5 else 0)
    if series_type == 'price':
        price = price[OCLH + ['label']]
    elif series_type == 'indicators':
        price = price[indicators + kdj + ['close', 'label']]
    # text = process_events(text, selected_elements =selected_elements)
    # text = text[['total']]
    text['date'] = pd.to_datetime(text['date'])
    text = text.sort_values(by='date', ascending=True).set_index('date')
    text = text[['news']]
    
    if concat_mode == 'inner':   
        text = text.groupby('date').transform(lambda x: ','.join(x)).drop_duplicates()
        con = text.merge(price, how='inner', left_index=True, right_index=True)# .iloc[price.shape[1]:, :]
    elif concat_mode == 'outer':
        con = text.merge(price, how='right', left_index=True, right_index=True)
    con = con.dropna()
    return con

def tokenize_nested_sequence(tokens, tokenizer, max_length=256):
    if isinstance(tokens, list):
        return [tokenize_nested_sequence(token, tokenizer, max_length) for token in tokens]
    else:
        return tokenizer(tokens, padding=True, truncation=True, max_length=max_length, return_tensors='pt')['input_ids']


