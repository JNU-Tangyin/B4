import os
MAIN_PATH = os.path.dirname(__file__)  # 当前文件所在的目录
os.chdir(MAIN_PATH)
print(f"Working in {MAIN_PATH}")
import torch
device = 'cpu' 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load_dataset
dataset_dir = 'dataset/'
dataset_type = 'USStock/' # 'CNStock/', 'USStock/', 'SP500'

# data_generator
WINDOW_SIZE = 20
LookBack = 20  # same as window_size
LookAhead = 1  # same as horizon; next-day directional label per paper
DATE_FORMAT = "%Y-%m-%d"
WEEKDAY = ['dayofweek']   # 注意这个是字符串，在和其他list相连的时候需要list + [WEEKDAY]，主要是用在对df[WEEKDAY]直接赋值
Symbol = '000001'
Period = 'daily'
StartDay = '20200101'
EndDay = '20240721'
MODEL_PATH  = 'mlp000001.pkl'

OCLHV = ['open',  'high', 'low', 'close', 'volume']
OCLH = ['open',  'high', 'low', 'close']
indicators = ['close_5_ema', 'close_10_ema','close_20_ema',"rsi_6", "rsi_12"]
kdj = ['kdjk', 'kdjd', 'kdjj']
# indicators = ['kdjk', 'kdjd', 'kdjj']
# indicators += ['boll','boll_lb','boll_ub']
# indicators += ["rsi_6", "rsi_12", "rsi_24"]
# indicators += ["macds","macdh","atr","vr","adx"] 
# indicators += ['close_5_ema', 'close_10_ema','close_20_ema','close_50_ema','close_100_ema']     # 有很多策略要用到很长周期的均线，若基于20日窗口就没法计算


# reprogram
patch_len = 16
stride = 8
d_model = 3
dropout = 0.1
num_tokens = 1000
n_heads = 2
d_ff = 32
d_llm = 768
num_classes = 2
vocab_size = 256
price_size = 3
hidden_size = 164

# dual
num_classes = 2
method = 'b4'
backend = False
alpha = 0.5
temp = 0.1
lr = 1e-5
decay = 0.01

# train
eps = 0.5
train_batch_size = 16
test_batch_size = 64
epoches = 1
retrain = True
action_nextday = False
cl_method = ['ce', 'scl', 'b4','infonce', 'bert', 'bert2', 'bert3']
baseline_method = ['stocknet', 'alstm', 'atten_lstm', 'transam', 'indexgan', 'dtml', 'taureau', 'espmp']

methods_list = ['b4']

stocknet_config = {
    'word_embed_size': 768,        # BERT 嵌入的尺寸
    'mel_h_size': 256,             # GRU 隐藏状态的尺寸
    'msg_embed_size': 100,         # 消息嵌入尺寸
    'g_size': 64,                  # 某一特定层的尺寸，可以根据需要调整
    'z_size': 32,                  # 变分隐变量的尺寸
    'price_embed_size': 3,         # 价格嵌入的尺寸
    'vocab_size': 256,           # BERT 模型的词汇表大小
    'output_size': 2               # 输出尺寸，可以删除 'y_size'
}

alstm_config = {'hidden_size': 30,
               'num_layers': 1,
               'output_size': 2,
               'word_embed_size': 768,
               'vocab_size': 256,
               'price_len':20, 'price_size':3}

transam_config = {'vocab_size':256,
               'word_embed_size': 768,
               'price_size': 3}

dtml_config = {'hidden_size': 10, 'vocab_size':256,
               'word_embed_size': 768,'price_len':20, 
               'price_size': 8,'output_size': 2,}

plot_param = {
    'id_vars': ['method', 'dataset', 'mask', 'budget'], # 聚合对象
    'value_vars': ['cer', 'wrc'], # 聚合后的值被操作的对象
    'save_type' : ".pdf",
    'columns' : ['method','dataset','metric','hp1','hp2','value'],
    'dis_pal' : "Set2", # 离散调色板
    'con_pal' : "Reds", # 连续调色板
    'tabs' : "tables/",
    'figs' : "figures/",
    'save_it' : True,
    'compared' : ['b4', 'stocknet', 'alstm', 'atten_lstm', 'transam'], # 对比实验绘制对象 ['Triviews','Stocknet']
    'ablation' : ['b4', 'ce', 'scl', 'infonce'] # 消融实验绘制对象

}



news_list = ['DE.csv','AWK.csv','COST.csv','FCX.csv',
 'BBL.csv','DUK.csv','WMT.csv','D.csv',
 'GE.csv','FB.csv','O.csv','ECL.csv',
 'DHR.csv','DLR.csv','ENB.csv','CAT.csv',
 'EXC.csv','CMCSA.csv','CHTR.csv','ADP.csv',
 'GOOG.csv','ADBE.csv','AAPL.csv','CSCO.csv',
 'BABA.csv','BP.csv','CVX.csv','BA.csv',
 'BRK-A.csv','ABT.csv','AVGO.csv','APD.csv',
 'COP.csv','AMZN.csv','DEO.csv','CCI.csv',
 'AMT.csv','ABBV.csv', 'BHP.csv','EL.csv',
 'ACN.csv','EQIX.csv','BAC.csv','EQNR.csv',
 'ACAD.csv','ADXS.csv','AES.csv','AGN.csv',
 'AL.csv','ALGN.csv','ALTR.csv','AM.csv',
 'AMTD.csv','ANET.csv','ARNA.csv','BAM.csv',
 'BBW.csv','BRO.csv','BWA.csv','CAG.csv',
 'CCI.csv','CINF.csv','CR.csv','CTAS.csv',
 'CYB.csv','CYBR.csv','DB.csv','DISCA.csv',
 'DLPH.csv','DRR.csv','DXJ.csv','ELY.csv',
 'EMN.csv','EQIX.csv','EWY.csv','FLT.csv',
 'GGG.csv','GPRO.csv','HEAR.csv','HMC.csv',
 'HRC.csv','HZNP.csv','INGN.csv','ITW.csv',
 'K.csv','KMB.csv','KOL.csv','LNC.csv',
 'M.csv','MDP.csv']


us_list = ['ABT.csv', 'AEP.csv', 'AMZN.csv', 'ASML.csv', 'BABA.csv',
 'CAT.csv', 'CMCSA.csv', 'D.csv', 'DIS.csv', 'ENB.csv',
 'EXC.csv', 'FB.csv', 'HON.csv','JD.csv',
 'JPM.csv','KO.csv','LLY.csv','LOW.csv','MA.csv',
 'MCD.csv','MMM.csv','MS.csv','NEE.csv','NEM.csv',
 'NFLX.csv','NGG.csv','NKE.csv','NVDA.csv','NVO.csv',
 'NVS.csv','ORCL.csv','PEP.csv','PFE.csv','PG.csv',
 'PLD.csv','PM.csv','PSA.csv','PTR.csv','PYPL.csv',
 'RIO.csv', 'RTX.csv', 'SBAC.csv', 'SBUX.csv', 'SCHW.csv',
 'SHW.csv', 'SNP.csv', 'SO.csv', 'SRE.csv', 'T.csv',
 'TGT.csv', 'TM.csv', 'TMO.csv', 'TMUS.csv', 'TSLA.csv',
 'TSM.csv', 'TTE.csv', 'UL.csv', 'UNH.csv', 'UPS.csv',
 'V.csv', 'VALE.csv', 'VZ.csv', 'WELL.csv', 'WMT.csv',
 'XEL.csv', 'XOM.csv', 'RDS-B.csv','JNJ.csv'] #'HD.csv',
