from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader as DataLoader
import pandas as pd
import numpy as np
import torch 
from globals import LookBack, indicators, WEEKDAY, OCLHV, OCLH
from functools import partial
from stockstats import StockDataFrame
from transformers import BertModel, BertTokenizer
import re, json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from utils import validate
# from OmniEvent.infer import infer


# def event_extraction(text):
#     results = infer(text=text, task="EE")
#     return results[0]["events"]


def seperate_event_element(df):
    '''
    Integrate event into a sequence, and decompose into three parts
    '''
    type_list = []
    trigger_list = []
    argument_list = []
    for index, event in enumerate(df.iterrows()):
        # print(event[1]['events'])
        data_str = event[1]['events'].replace("'", '"')
        data = json.loads(data_str)
        # print(data)
        type_list.append([item['type'] for item in data])
        trigger_list.append([item['trigger'] for item in data])
        argument_list.append([item['arguments'] for item in data])
        
    df['type'] = type_list
    df['trigger'] = trigger_list
    df['arguments'] = argument_list
    return df 


def argument_process(df):
    '''
    Process the 'arguments' column in the DataFrame, extracting the 'mention' from each element and storing it in a list.
    '''
    mention_list = []
    for i in df['arguments']:
        m_list=[]
        # print(i)
        for m in i[0]:
            # print(m)
            m_list.append(m['mention'])
        mention_list.append(m_list)
    df['arguments'] = mention_list
    return df


def convert(string):
    '''Extracts content within square brackets from a given string.
    '''
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, string)
    if match:
        result = match.group(1).replace("'", '')
        return result
    else:
        print('未找到')
        
        
def shift_datetime(date, shift_from='%Y%m%d', shift_to='%Y%m%d'):
    date = pd.to_datetime(date, format=shift_from)
    date = date.dt.strftime(shift_to)
    return date


def get_embeddings(text, tokenizer, model):
    '''
    Get the embeddings for the given text using the provided tokenizer and model.
    '''
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    outputs = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return outputs


def get_embeddings_sum(embedding):
    return sum(embedding)/len(embedding)

   
def process_events(text_df, selected_elements=['type']):
    df = text_df.copy()
    # df['date'] = df['date'].shift(-1)
    # df = df.dropna()
    # df = df[df['events']!='[]']
    # first_col = df.pop('date')
    # df.insert(0, 'date', first_col)
    
    df = seperate_event_element(df)
    df = argument_process(df)
    df[selected_elements] = df[selected_elements].applymap(lambda x:convert(str(x)))
    df['total'] = df[selected_elements[0]]  
    for i in selected_elements[1:]:
        df['total'] += df[i]    
    df['date'] = shift_datetime(df['date'], shift_from='%Y-%m-%d', shift_to='%Y-%m-%d')
    df = df.sort_values(by='date', ascending=True).set_index('date')
    return df[selected_elements + ['total']]


def event_embedding_sum(events, tokenizer, model):
    events = events.applymap(get_embeddings, tokenizer=tokenizer, model=model)
    events = events.applymap(get_embeddings_sum)
    return events


def event_embedding(events, tokenizer, model):
    events['embedding'] = events['total'].apply(lambda x: get_embeddings(x, tokenizer=tokenizer, model=model))
    events['embedding'] = events['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    embeddings_array = np.vstack(events['embedding'].values)

    for i in range(embeddings_array.shape[1]):
        events['column' + str(i)] = 0
    events.iloc[:, -embeddings_array.shape[1]:] = embeddings_array
    return events.iloc[:, -embeddings_array.shape[1]:]

# if __name__ == '__main__':
#     from load_data import *
#     from transformers import AutoTokenizer, AutoModel
#     from OmniEvent.infer import infer