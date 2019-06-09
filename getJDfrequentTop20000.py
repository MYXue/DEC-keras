# -*- coding: utf-8 -*-
# 找到评论数据中的高频20000词
import pandas as pd
import jieba
from collections import Counter
import pickle

path='jd_Data/extract_comments4.txt'
columns = ['text','stars']
df = pd.read_csv(path,header=None,sep='\t',names=columns) 
df.dropna(axis=0, how='any', inplace=True) #删除有空值的行

frequentTop20000 = []

jieba.load_userdict("jd_Data/user_dict.txt")

c = Counter()

for text in df['text']:
    seg_list = jieba.cut(text)
    for x in seg_list:
        c[x] += 1

for (k,v) in c.most_common(20000):
    frequentTop20000.append(k)

print (frequentTop20000)

with open('jd_Data/frequentTop20000.pk', 'wb') as f:
    pickle.dump(frequentTop20000, f)