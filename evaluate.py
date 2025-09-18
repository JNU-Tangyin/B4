import pandas as pd
import numpy as np
from stockstats import StockDataFrame
# from utils import validate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_context("notebook")
import os


class Evaluator():
    def __init__(self, sdf:StockDataFrame, dataset_type, name):
        # validate(sdf, required=['close', 'action'])
        self.sdf = sdf
        self.result = self.evaluate()   # 业绩序列
        self.name = name
        self.dataset_type = dataset_type
        self.fig = self.plot()
        # 做空的日期
    
    def evaluate(self) -> pd.DataFrame:
        sdf = self.sdf.copy()
        # 有些策略有可能会留下0，这里不考虑hold的情况
        sdf['action'] = sdf['action'].replace(0,np.nan).ffill()  # hold的内涵：前面为buy时我继续buy，前面sell时我继续sell
        sdf['pctchg'] = sdf.close.pct_change().fillna(0)
        
        ## 计算B&H
        underline = sdf.close/sdf.close.iloc[0]            # 将close归1
        ## 1x
        long1x = sdf.pctchg[sdf.action == 1] + 1            # 取移动平均收益率>1的部分
        short1x = (sdf.pctchg[sdf.action == -1] + 1)**(-1)  # 取移动平均收益率<1的部分，做空
        flat1x = (sdf.pctchg[sdf.action == -1] + 1)**(0)      # 取移动平均收益率<1的部分，不做空，即全赋值为1
        do_short = pd.concat([long1x,short1x], axis=0).sort_index().cumprod()   # 拼接两部分，做空
        no_short = pd.concat([long1x,flat1x], axis=0).sort_index().cumprod()      # 拼接两部分，不做空
        

        ## 2x
        long2x = 2*sdf.pctchg[sdf.action == 1] + 1              # 取移动平均收益率>1的部分，2倍做多
        short2x = (2*sdf.pctchg[sdf.action == -1] + 1)**(-1)    # 取移动平均收益率<1的部分，2倍做空
        flat2x = (2*sdf.pctchg[sdf.action == -1] + 1)**(0)      # 取移动平均收益率<1的部分，不做空，即全赋值为1
        ETF2x_short = pd.concat([long2x,short2x], axis=0).sort_index().cumprod()    # 拼接两部分，2倍做空
        ETF2x_no_short = pd.concat([long2x,flat2x], axis=0).sort_index().cumprod()  # 拼接两部分，2倍不做空

        df = pd.DataFrame({"B&H": underline,
                        "1x_short":do_short, 
                        "1x_no_short":no_short,
                        "2x_short":ETF2x_short,
                        "2x_no_short":ETF2x_no_short,
                        "action":sdf.action,
                        "labels": sdf.labels})

            
        ## 输出最后一行作为观察
        print(df.tail(1))
        self.result = df
        return self.result
    
    def collect_shorts(self):
        sdf = self.result.copy()
        sdf['norm_close'] = sdf['B&H']/sdf['B&H'].iloc[0]
        sdf['last_action'] = sdf.action.shift(1).fillna(0)    # compared with yesterday
        action_days = sdf[sdf.action != sdf.last_action] 
        self.action_days = action_days
        action_days = action_days.reset_index() # 将index恢复为column
        action_days['next_day'] = action_days.shift(-1).date
        action_days['next_day'] = action_days['next_day'].fillna(sdf.index[-1]) # 用d最后一行的date补齐最后一行的空值
        self.short_days = action_days[action_days.action == -1][['date','next_day']]
        
    def attach_grey(self, axes):

        def plot_range(x, x_axis, floor, ceiling):
            ax.fill_between(x_axis, floor, ceiling, (x.date <= x_axis) & (x_axis< x.next_day), color = "k", alpha = 0.05)
        
        if isinstance(axes, mpl.axes._axes.Axes):   # if it is a single instance other than an array of them.
            axes = [axes] 
        
        short_days = self.short_days.copy()       
        short_days = short_days.reset_index(drop=True)   # index恢复成`date`，plot_range中要用到

        for ax in axes:
            floor, ceiling = ax.get_ylim()
            short_days.apply(plot_range, args=(self.sdf.index, floor, ceiling), axis=1) 
            ax.set_xticks(pd.concat([short_days.date, short_days.next_day], axis=0).sort_index())
            ax.tick_params(axis='x', labelsize=6)
    
    def attach_grey2(self, axes):
        def plot_range(x, x_axis, floor, ceiling):
            if len(x) < 2:
                return 
            day1 = x.index[0]
            day2 = x.index[1]
            action1 = x[0]
            print(day1, day2, action1)
            ax.fill_between(x_axis, floor, ceiling, 
                            (day1 <= x_axis) & (x_axis< day2), 
                            color = "k" if action1 == -1 else "w", 
                            alpha = 0.05
                            )
        def plot_range2(x, x_axis, floor, ceiling):
            pass
        
        if isinstance(axes, mpl.axes._axes.Axes):   # if it is a single instance other than an array of them.
            axes = [axes] 
        
        for ax in axes:
            floor, ceiling = ax.get_ylim()
            self.action_days.action.dropna().rolling(2, min_periods=2).apply(plot_range2, args=(self.sdf.index, floor, ceiling)) 
            ax.set_xticks(self.action_days.index)
            ax.tick_params(axis='x', labelsize=6) 

    def plot(self): #**kwargs
        fig, axes = plt.subplots(2,1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, dpi=120)
        self.result[['1x_short', '1x_no_short','2x_short', '2x_no_short']].plot(ax=axes[0], colormap="viridis",title= self.dataset_type + self.name)
        self.result["B&H"].plot(ax=axes[1], rot=90, title= self.dataset_type[:-1]+ self.name)
        self.collect_shorts()
        self.attach_grey(axes)
        self.fig = fig
        return fig 
    

    
    def cal_sharpe(self):
        risk_free_rate = 0.03  
        sharpe_ratios = {
            "sharpe_1x_short": (self.result["1x_short"].pct_change().mean() - risk_free_rate) / self.result["1x_short"].pct_change().std(),
            "sharpe_1x_no_short": (self.result["1x_no_short"].pct_change().mean() - risk_free_rate) / self.result["1x_no_short"].pct_change().std(),
            "sharpe_2x_short": (self.result["2x_short"].pct_change().mean() - risk_free_rate) / self.result["2x_short"].pct_change().std(),
            "sharpe_2x_no_short": (self.result["2x_no_short"].pct_change().mean() - risk_free_rate) / self.result["2x_no_short"].pct_change().std()
        }
        self.total_result = self.total_result.assign(**sharpe_ratios)  

        return self.total_result
    
    def cal_mdd(self):
        self.result['daily_return'] = self.sdf['close'].pct_change()
        self.result['cumulative_return'] = (1 + self.result['daily_return']).cumprod()
        self.result['cumulative_return_max'] = self.result['cumulative_return'].cummax()
        self.result['drawdown'] = self.result['cumulative_return_max'] - self.result['cumulative_return']
        self.total_result['max_drawdown'] = self.result['drawdown'].max()
    
        return self.total_result
    
    def cal_acc(self):
        correct_predictions = self.result[(self.result['action'] == self.result['labels'])].shape[0]
        total_predictions = self.result.shape[0]
        accuracy = correct_predictions / total_predictions
        self.total_result['acc'] = accuracy
        return self.total_result

    
    def check_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save(self, method, eps, epoches, lookahead, lookback, freeze_ornot, series_type, text_type, alpha, temp,losspull):
        # param = f'{method}_tune({freeze_ornot})_series({series_type})_text({text_type})_lookahead{lookahead}_lookback{lookback}_alpha{alpha}_temp{temp}_losspull{losspull}_epoches{epoches}'
        param = f'method_{method}_eps_{eps}_epoches_{epoches}'
        result_file_path = f'result/{self.dataset_type}/{param}/table/{self.name}'
        fig_file_path = f'result/{self.dataset_type}/{param}/pic/{self.name[:-4]}.pdf'

        result_directory = os.path.dirname(result_file_path)
        fig_directory = os.path.dirname(fig_file_path)

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        if not os.path.exists(fig_directory):
            os.makedirs(fig_directory)

        self.result.to_csv(result_file_path, index=False)
        self.fig.savefig(fig_file_path)
        
        self.total_result = self.result.tail(1)
        self.total_result = self.cal_sharpe()
        self.total_result = self.cal_mdd()
        self.total_result = self.cal_acc()
        
        return self.total_result