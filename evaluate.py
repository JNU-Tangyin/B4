import pandas as pd
import numpy as np
from stockstats import StockDataFrame
# from utils import validate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_context("notebook")
import os


# Trading protocol constants per paper
TC_PER_SIDE = 0.0005       # 5 bps transaction cost per side
SLIPPAGE_PER_SIDE = 0.0005 # 5 bps slippage per side
TOTAL_COST_PER_TRADE = TC_PER_SIDE + SLIPPAGE_PER_SIDE  # 10 bps per side
RISK_FREE_RATE = 0.03 / 252  # daily risk-free rate (3% annualised)


class Evaluator():
    def __init__(self, sdf:StockDataFrame, dataset_type, name):
        # validate(sdf, required=['close', 'action'])
        self.sdf = sdf
        self.name = name
        self.dataset_type = dataset_type
        self.result = self.evaluate()
        self.fig = self.plot()
    
    def evaluate(self) -> pd.DataFrame:
        sdf = self.sdf.copy()
        sdf['pctchg'] = sdf.close.pct_change().fillna(0)
        
        # Long-flat protocol: action in {1=long, 0=flat}
        # No forward-fill needed because the model emits an explicit signal each day.
        sdf['position'] = sdf['action']
        
        # Identify position changes (flat->long or long->flat) to deduct costs
        sdf['position_change'] = (sdf['position'] != sdf['position'].shift(1)).astype(int)
        
        # Strategy gross return = position * market return
        sdf['strategy_gross_ret'] = sdf['position'] * sdf['pctchg']
        # Deduct per-side transaction cost + slippage whenever the position changes
        sdf['cost'] = sdf['position_change'] * TOTAL_COST_PER_TRADE
        sdf['strategy_net_ret'] = sdf['strategy_gross_ret'] - sdf['cost']
        
        # Buy-and-Hold baseline (no turnover costs)
        sdf['bh_ret'] = sdf['pctchg']
        
        # Cumulative NAVs
        sdf['strategy_gross_nav'] = (1 + sdf['strategy_gross_ret']).cumprod()
        sdf['strategy_net_nav']   = (1 + sdf['strategy_net_ret']).cumprod()
        sdf['bh_nav']             = (1 + sdf['bh_ret']).cumprod()
        
        df = pd.DataFrame({
            "B&H": sdf['bh_nav'],
            "Strategy_gross": sdf['strategy_gross_nav'],
            "Strategy_net": sdf['strategy_net_nav'],
            "position": sdf['position'],
            "position_change": sdf['position_change'],
            "strategy_gross_ret": sdf['strategy_gross_ret'],
            "strategy_net_ret": sdf['strategy_net_ret'],
            "labels": sdf.labels
        })
        
        self.result = df
        return self.result
    
    def cal_sharpe(self):
        """Annualised Sharpe ratio using net daily returns."""
        net_rets = self.result['strategy_net_ret'].dropna()
        if net_rets.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (net_rets.mean() - RISK_FREE_RATE) / net_rets.std() * np.sqrt(252)
        self.total_result['sharpe'] = sharpe
        return self.total_result
    
    def cal_sortino(self):
        """Annualised Sortino ratio using net daily returns."""
        net_rets = self.result['strategy_net_ret'].dropna()
        downside = net_rets[net_rets < RISK_FREE_RATE]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        if downside_std == 0:
            sortino = 0.0
        else:
            sortino = (net_rets.mean() - RISK_FREE_RATE) / downside_std * np.sqrt(252)
        self.total_result['sortino'] = sortino
        return self.total_result
    
    def cal_turnover(self):
        """Turnover = sum of absolute position changes / number of days."""
        turnover = self.result['position_change'].sum() / len(self.result)
        self.total_result['turnover'] = turnover
        return self.total_result
    
    def cal_mdd(self):
        nav = self.result['Strategy_net']
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        self.total_result['max_drawdown'] = drawdown.min()
        return self.total_result
    
    def cal_acc(self):
        correct = (self.result['position'] == self.result['labels']).sum()
        self.total_result['acc'] = correct / len(self.result)
        return self.total_result
    
    def cal_factor_alpha(self, factors_df=None):
        """
        Factor-adjusted alpha.  Requires daily factor returns (Mkt-RF, SMB, HML, MOM, RF).
        Pass a DataFrame with columns ['Mkt_RF','SMB','HML','MOM','RF'] aligned by date.
        If not provided, alpha is left as NaN.
        """
        if factors_df is None:
            self.total_result['ff3_alpha'] = np.nan
            self.total_result['carhart_alpha'] = np.nan
            return self.total_result
        
        merged = self.result[['strategy_net_ret']].join(factors_df, how='inner')
        merged['excess_ret'] = merged['strategy_net_ret'] - merged['RF']
        
        # FF3 regression
        try:
            import statsmodels.api as sm
            X3 = sm.add_constant(merged[['Mkt_RF','SMB','HML']])
            model3 = sm.OLS(merged['excess_ret'], X3).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            self.total_result['ff3_alpha'] = model3.params['const']
            self.total_result['ff3_alpha_t'] = model3.tvalues['const']
        except Exception:
            self.total_result['ff3_alpha'] = np.nan
            self.total_result['ff3_alpha_t'] = np.nan
        
        # Carhart 4-factor regression
        try:
            X4 = sm.add_constant(merged[['Mkt_RF','SMB','HML','MOM']])
            model4 = sm.OLS(merged['excess_ret'], X4).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            self.total_result['carhart_alpha'] = model4.params['const']
            self.total_result['carhart_alpha_t'] = model4.tvalues['const']
        except Exception:
            self.total_result['carhart_alpha'] = np.nan
            self.total_result['carhart_alpha_t'] = np.nan
        
        return self.total_result
    
    def plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]}, dpi=120)
        self.result[['B&H', 'Strategy_gross', 'Strategy_net']].plot(
            ax=axes[0], colormap="viridis", title=self.dataset_type + self.name)
        self.result['position'].plot(ax=axes[1], rot=90, drawstyle='steps-post',
                                      title='Position (1=long, 0=flat)')
        self.fig = fig
        return fig 
    

    
    def save(self, method, eps, epoches, lookahead, lookback, freeze_ornot, series_type, text_type, alpha, temp, losspull, factors_df=None):
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
        
        self.total_result = self.result.tail(1).copy()
        self.total_result = self.cal_sharpe()
        self.total_result = self.cal_sortino()
        self.total_result = self.cal_turnover()
        self.total_result = self.cal_mdd()
        self.total_result = self.cal_acc()
        self.total_result = self.cal_factor_alpha(factors_df)
        
        return self.total_result