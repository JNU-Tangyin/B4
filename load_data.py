import pandas as pd
import chardet

# dataset_dir = 'carbon/'
# name = 'guangzhou'

def load_news(dataset_dir='dataset/', dataset_type='CNStock/',  name='000001'):
    text_type='news'
    if dataset_type =='USStock/':
        news = pd.read_csv(f'{dataset_dir}{dataset_type}{text_type}/{name}.csv', delimiter=',')
    else:
        news = pd.read_csv(f'{dataset_dir}{dataset_type}{text_type}/{name}.csv')
    return news


def load_events(dataset_dir='dataset/', dataset_type='CNStock/',  name='000001'):
    text_type='events'
    event = pd.read_csv(f'{dataset_dir}{dataset_type}{text_type}/{name}.csv')
    return event


def load_price(dataset_dir='dataset/', dataset_type='CNStock/',  name='000001'):
    text_type='price'
    price = pd.read_csv(f'{dataset_dir}{dataset_type}{text_type}/{name}.csv')
    return price