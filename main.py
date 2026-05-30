from generator.generator import data_generator

from exp.exp_triviews import *
from exp.exp_baseline import *
from transformers import BertModel
from evaluate import Evaluator
from globals import *
import os
import pandas as pd
import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
random.seed(42)

def main(news_list, dataset_dir = 'dataset/', dataset_type = 'CNStock/', action_nextday = False, eps=0.5,  retrain = True, epoches=200, 
         lookahead=LookAhead, lookback = LookBack, freeze_ornot=True, series_type='price', text_type='news', \
             alpha=0.5, temp=0.1, losspull=0):
    news_path = dataset_dir + dataset_type +'news/'
    start = 0
    end = 1
    # news_list = os.listdir(news_path)[start:end]
    # news_list = ['A.csv']
    
    # 消融时bert担任编码和决策的角色，循环前完成初始化，对比时担任决策角色
    if dataset_type == 'CNStock/': 
        bert = BertModel.from_pretrained('bert-chinese').to(device)
    else: bert = BertModel.from_pretrained('bert-uncased').to(device)
    result = []
    for name in news_list:
        for method in methods_list:
            # 判断result中是否已有结果
            result_path = os.path.join('result/', \
            dataset_type, f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_alpha{alpha}_temp{temp}_losspull{losspull}_epoches{epoches}/table/')
            if not os.path.exists(result_path): 
                os.makedirs(result_path)
                print(f'{result_path} have created')
            else:
                result_list = os.listdir(result_path)
                if name in result_list: 
                    print(f'{name} have existed in {result_path}')
                    continue
            n = name[:-4]
            print(f'training the {n} in {method}...')
            train_df, test_df, train_dataloader, test_dataloader = data_generator(\
            dataset_dir = dataset_dir, dataset_type =dataset_type, name=n, method=method, \
                lookahead=lookahead, lookback=lookback, series_type=series_type, text_type=text_type)
            if method in cl_method:
                test_df = test_triviews(train_df, test_df, train_dataloader, test_dataloader,\
                action_nextday=action_nextday, eps=eps,  retrain=retrain, epoches=epoches, method=method,\
                    dataset_dir=dataset_dir, dataset_type=dataset_type, name=n,
                    lookahead=lookahead, lookback=lookback, freeze_ornot=freeze_ornot, losspull=losspull)
            elif method in baseline_method:
                test_df = test_baseline(train_df, test_df, train_dataloader, test_dataloader, bert,\
                action_nextday=action_nextday, eps=eps,  retrain=retrain, epoches=epoches, method=method,\
                    dataset_dir=dataset_dir, dataset_type=dataset_type, name=n)      
            sig_result = Evaluator(test_df, dataset_type, name).save(method, eps, epoches, lookahead, lookback, \
                freeze_ornot, series_type, text_type, alpha, temp, losspull)   
            result_dict = {'name':name,
                           'method':method,
                           'B&H': sig_result['B&H'].values[0],
                           'strategy_net': sig_result['Strategy_net'].values[0],
                           'sharpe': sig_result['sharpe'].values[0],
                           'sortino': sig_result['sortino'].values[0],
                           'turnover': sig_result['turnover'].values[0],
                           'max_drawdown': sig_result['max_drawdown'].values[0],
                           'acc': sig_result['acc'].values[0],
                           'ff3_alpha': sig_result.get('ff3_alpha', np.nan),
                           'carhart_alpha': sig_result.get('carhart_alpha', np.nan)}
            
            result.append(result_dict)
            
    result_df = pd.DataFrame(result)
    file_path = f'result/{dataset_type}{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_alpha{alpha}_temp{temp}_losspull{losspull}_epoches{epoches}.csv'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # result_df.to_csv(file_path, index=False)
    print(f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_alpha{alpha}_temp{temp}_losspull{losspull}_epoches{epoches}')


    
if __name__ == '__main__':
    from globals import *
    import itertools
    epoches_list = [200]
    lookahead_list = [2]
    freeze_ornot_list = [True]
    alpha_list = [0.9]  # 0.3, 0.7
    temp_list = [0.1]  # 0.1, 0.3, 0.5, 0.7, 0.9
    losspull_list = [0, 1, 2, 3]  # causal backward window Delta (paper: strictly past j < i)
    
    # for params in itertools.product(epoches_list, lookahead_list, freeze_ornot_list, alpha_list, temp_list, losspull_list):
    #     epoches, lookahead, freeze_ornot, alpha, temp, losspull = params
    #     main(us_list, dataset_type='USStock/', epoches=epoches, lookahead=lookahead, freeze_ornot=freeze_ornot, series_type='price', text_type='news', alpha=alpha, temp=temp, losspull=losspull)
        
    # main(cn_list, dataset_type='CNStock/', epoches=30, lookahead=2, freeze_ornot=True, series_type='price', text_type='news', alpha=0.5, temp=0.1, losspull='±1')    
    main(us_list, dataset_type='USStock/', epoches=1, lookahead=2, freeze_ornot=True, series_type='price', text_type='news', alpha=0.5, temp=0.1, losspull=0)  
    # main(sp_list, dataset_type='SP500/', epoches=30, lookahead=1, freeze_ornot=True, series_type='price', text_type='news', alpha=0.5, temp=0.1, losspull=0)     