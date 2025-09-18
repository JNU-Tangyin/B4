import torch
import torch.nn as nn
from textblob import TextBlob
import pandas as pd
import re
from globals import device
import numpy as np
from snownlp import SnowNLP

class Taureau(nn.Module):
    def __init__(self, dataset_type):
        super(Taureau, self).__init__()
        self.language = dataset_type[:2]

    def getSentimentPolarity(self, x):
        testimonial=TextBlob(x)
        s=testimonial.sentiment.polarity
        return s
    
    def getSentimentSubjectivity(self, x):
        testimonial=TextBlob(x)
        s=testimonial.sentiment.subjectivity
        return s
    
    def snownlp(self, x):
        s = SnowNLP(x)
        return s.sentiments
    
        
    def forward(self, news, price):
        news = np.array(news)[:,-1]
        if self.language == 'US':
            polarity = np.vectorize(self.getSentimentPolarity)(news)
            senti_score = polarity
            outputs = [-1 if x < 0 else 1 for x in senti_score]
            # polarity = news.map(self.getSentimentPolarity)
            # subjectivity = news.map(self.getSentimentPolarity)
            # aggregate_score = polarity*0.5+subjectivity*0.5
        else:
            senti_score = np.vectorize(self.snownlp)(news)
            outputs = [-1 if x < 0.5 else 1 for x in senti_score]
        return outputs
