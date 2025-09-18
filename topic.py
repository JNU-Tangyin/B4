import os
import pandas as pd
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from bertopic import BERTopic

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

stock_list = os.listdir('dataset/USStock/news')


stock_list = [i[:-4] for i in stock_list]
for stock in stock_list:
     file_path = os.path.join('topic', f'{stock}_topic.csv')
     if os.path.exists(file_path):
        print(f'{stock}_topic.csv 文件已存在，跳过...')
     else:
         print(f'{stock}_topic.csv 文件不存在，开始处理...')
         docs = pd.read_csv(f'dataset/USStock/news/{stock}.csv')['news']  #[:100].values
         topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, 
               nr_topics=10,top_n_words = 20)  #diversity=0.5,
         topics, probabilities = topic_model.fit_transform(docs)
         print(topic_model.get_topic_info())
         topic_model.get_topic_info().to_csv(f'topic/{stock}_topic.csv')
    
# doc_df = pd.DataFrame()
# for stock in stock_list:
#     docs = pd.read_csv('dataset/USStock/news/'+stock)['news']
#     doc_df = pd.concat([doc_df, docs])
# doc_df = doc_df.dropna()
# topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, 
#       nr_topics=10,top_n_words = 20)  #diversity=0.5,
# topics, probabilities = topic_model.fit_transform(doc_df.iloc[:,0].values)
# topic_model.get_topic_info().to_csv(f'all_topic.csv')

