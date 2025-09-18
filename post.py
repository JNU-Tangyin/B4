import pandas as pd
import os
from globals import news_list
import numpy as np

new_dir = '/data/JupyterLab/xiaotong/trading_strategy4.0/'
os.chdir(new_dir)

def get_file_lists(dataset, method, base_path):
    stock_list = os.listdir(base_path.format(dataset=dataset, method=method))
    return stock_list

def calculate_metric(dataset, method, name, base_path, risk_free_rate = 0):
    df = pd.read_csv(base_path.format(dataset=dataset, method=method)+name)
    
    df['close'] = df['B&H']*df['B&H'][0]
    daily_return = df['1x_no_short'].pct_change()
    # cumulative_return = (1 + daily_return).cumprod()
    cumulative_return = df['1x_no_short']
    cumulative_return_max = cumulative_return.cummax()
    drawdown = cumulative_return_max - cumulative_return
    
    correct_predictions = df[(df['action'] == df['labels'])].shape[0]
    total_predictions = df.shape[0]
    accuracy = correct_predictions / total_predictions
    df['acc'] = accuracy

    sharpe_ratios = {
        "sharpe_1x_short": ((df["1x_short"] -1).mean() - risk_free_rate) / df["1x_short"].std(),
        "sharpe_1x_no_short": ((df["1x_no_short"] -1).mean() - risk_free_rate) / df["1x_no_short"].std(),
        "sharpe_2x_short": (df["2x_short"] -1 - risk_free_rate) / df["2x_short"].std(),
        "sharpe_2x_no_short": (df["2x_no_short"] -1 - risk_free_rate) / df["2x_no_short"].std()
    }

    annual_return = cumulative_return.iloc[-1] ** (252 / len(daily_return)) - 1
    # annual_return = (cumulative_return.iloc[-1] - 1) ** (252 / len(daily_return))
    cumulative_returns = cumulative_return.iloc[-1] - 1
    annual_volatility = daily_return.std() * np.sqrt(252)
    calmar_ratio = annual_return / abs(drawdown.max())
    stability = daily_return.corr(pd.Series(range(len(daily_return))))
    max_drawdown = drawdown.max()
    omega_ratio = (daily_return[daily_return > 0].sum() / abs(daily_return[daily_return < 0].sum())) if daily_return[daily_return < 0].sum() != 0 else np.nan
    sortino_ratio = (daily_return.mean() - risk_free_rate) / daily_return[daily_return < 0].std()
    skew = daily_return.skew()
    kurtosis = daily_return.kurt()
    tail_ratio = abs(daily_return[daily_return > 0].mean() / daily_return[daily_return < 0].mean())
    daily_value_at_risk = np.percentile(daily_return.dropna(), 5)
    alpha = (annual_return - risk_free_rate) / annual_volatility
    beta = daily_return.cov(pd.Series(range(len(daily_return)))) / pd.Series(range(len(daily_return))).var()

    # 结果汇总
    result = df.tail(1)
    result.insert(0, 'code', str(name[:-4]))
    result.insert(1, 'method', method)
    result.insert(2, 'sharpe_1x_short', sharpe_ratios['sharpe_1x_short'])
    result.insert(3, 'sharpe_1x_no_short', sharpe_ratios['sharpe_1x_no_short'])
    result.insert(4, 'sharpe_2x_short', sharpe_ratios['sharpe_2x_short'])
    result.insert(5, 'sharpe_2x_no_short', sharpe_ratios['sharpe_2x_no_short'])
    result.insert(6, 'annual_return', annual_return)
    result.insert(7, 'cumulative_returns', cumulative_returns)
    result.insert(8, 'annual_volatility', annual_volatility)
    result.insert(9, 'calmar_ratio', calmar_ratio)
    result.insert(10, 'stability', stability)
    result.insert(11, 'max_drawdown', max_drawdown)
    result.insert(12, 'omega_ratio', omega_ratio)
    result.insert(13, 'sortino_ratio', sortino_ratio)
    result.insert(14, 'skew', skew)
    result.insert(15, 'kurtosis', kurtosis)
    result.insert(16, 'tail_ratio', tail_ratio)
    result.insert(17, 'daily_value_at_risk', daily_value_at_risk)
    result.insert(18, 'alpha', alpha)
    result.insert(19, 'beta', beta)
    result.insert(20, 'accuracy', accuracy)
    result.insert(0, 'dataset', dataset)
    return result


def wide_to_long(df, tab_param):
    df['method'] = pd.Categorical(df['method'], categories=tab_param['method_sort'], ordered=True)
    df['dataset'] = pd.Categorical(df['dataset'], categories=tab_param['datasets_sort'], ordered=True)
    
    result_df = pd.melt(df, id_vars=plot_param['id_vars'], value_vars=plot_param['value_vars'], var_name='metric', value_name='value')
    result_df = result_df[plot_param['id_vars'] + ['metric', 'value']]
    result_df = result_df.sort_values(by=plot_param['id_vars']).reset_index().drop(columns='index')
    return result_df

def table_to_tex(df, tab_param):
    df = wide_to_long(df, tab_param)
    dd = df[tab_param['table_index'] + tab_param['table_columns'] + tab_param['table_values']]
    dd = dd.round(3) # 小数点后3位够了
    table = dd.pivot_table(index=tab_param['table_index'], columns=tab_param['table_columns'], values=tab_param['table_values'])
    table.to_latex(
            # buf=tabs+"{}_".format(tab_param['tab_name']) +percent+".tex",
            buf=tab_param['path']+"{}".format(tab_param['tab_name']) +".tex", # 文件名有标识
            # caption="Comparative performance with {}\% masked".format(int(masked*100)), # 标题
            label='label',  # label
            float_format = tab_param['float_format'],    # 表格中的float，小数点后2位数
            position = tab_param['position'],        # 表格的位置
            column_format = tab_param['column_format'], # 表格列的对齐方式
            escape = True           # 对latex敏感的字段做escape处理，即在前面加"\"
            )
    return table

if __name__ ==  '__main__':
    # plot_param and tab_param 的dataset要一起改
    from globals import cn_list, us_list, sp_list
    dataset_to_list = {'CNStock': cn_list, 'USStock': us_list, 'SP500': sp_list}
    value_var = ['Accuracy','Sharpe Ratio','Max Drawdown','Annual Return', 'Cumulative Returns',
        'Annual Volatility', 'Calmar Ratio','Stability', 'Omega Ratio','Sortino Ratio',
        'Skew', 'Kurtosis', 'Tail Ratio', 'VaR', 'Alpha','Beta']
    
    # selected metric
    # value_var = ['Accuracy','Sharpe Ratio','Max Drawdown','Annual Return', 'Cumulative Returns',
    #     'Annual Volatility','VaR', 'Alpha','Beta']    
    plot_param = {
                # 'id_vars':['code', 'metric'],                              
                  'id_vars':['dataset', 'method'],   
            #   'value_vars':['Triviews', 'LSTM', 'Attentive-LSTM', 'StockNet', 'Transformer', 'ESPMP','DUAL-DNN','SCL-DNN'],
              'value_vars': value_var,
                            }
    # 记得改文件名字tab_name
    tab_param = {'tab_name':'整体股', 'path':'latex/',
             'table_index': ['dataset', 'metric'], 'table_columns':['method'], 
            #  'table_index': ['code','method'], 'table_columns':['metric'],    # table_index=['code'] ['dataset']
             'table_values': ['value'], 'float_format':"%.3f", 'position':"htbp",
             'column_format': "l|l|cc|cc|cc|cc|cc|cc|cc|cc|cc|cc",
             'method_sort': ['B4', 'LSTM', 'Attentive-LSTM', 'StockNet', 'Transformer', 'ESPMP','DUAL-DNN','SCL-DNN'],
            #  'method_sort': ['B4(dual_cl)', 'B4(supcon_cl)', 'B4(no_cl)','B4(not_translated)'],
             'datasets_sort': ['CNStock', 'USStock', 'SP500']}

    # 选择方法，后面要换新名字
    methods = ['b4', 'alstm', 'atten_lstm', 'stocknet', 'transam', 'infonce', 'bert2', 'bert3']
    # methods = ['b4', 'scl', 'ce','bert']
    
    datasets = ['CNStock', 'USStock', 'SP500']
    base_path = 'result/{dataset}/method_{method}_eps_0.5_epoches_200/table/'
    result_list = []
    for dataset in datasets:
        for method in methods:
            # stock_list = get_file_lists(dataset, method, base_path)
            stock_list = dataset_to_list[dataset]
            for stock in stock_list:
                result = calculate_metric(dataset, method, stock, base_path)
                result_list.append(result)
    final_result = pd.concat(result_list, ignore_index=True).sort_values('code')
    final_result = final_result.rename(columns={'accuracy': 'Accuracy',
        'sharpe_1x_no_short':'Sharpe Ratio','max_drawdown':'Max Drawdown',
        'annual_return': 'Annual Return', 'cumulative_returns':'Cumulative Returns',
        'annual_volatility':'Annual Volatility', 'calmar_ratio':'Calmar Ratio',
        'stability':'Stability', 'omega_ratio':'Omega Ratio','sortino_ratio':'Sortino Ratio',
        'skew':'Skew', 'kurtosis':'Kurtosis', 'tail_ratio':'Tail Ratio',  
        'daily_value_at_risk': 'VaR', 'alpha':'Alpha','beta':'Beta',
        })  # all
    
    
    new_method_name = ['Triviews', 'LSTM', 'Attentive-LSTM', 'StockNet', 'Transformer', 'ESPMP','DUAL-DNN','SCL-DNN']
    # new_method_name = ['Triviews(dual_cl)', 'Triviews(supcon_cl)', 'Triviews(no_cl)','Triviews(not_translated)'] 

    # 新方法命名
    for i in range(len(new_method_name)):
        final_result.loc[final_result['method']== methods[i], 'method'] = new_method_name[i]
    
    final_result.to_csv('result/final3.csv', index=False)
    table_to_tex(final_result, tab_param)