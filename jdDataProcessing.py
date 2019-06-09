# -*- coding: utf-8 -*-
'''
读取京东评论数据，分词，转化评分等级
'''
import pandas as pd
import numpy as np
from tqdm import tqdm #一个快速、可扩展的Python进度条
import jieba
import re
import pickle
# import sklearn

STOP_WORDS = [' '] #停用词

def polarize(v): #如果想把1-5分打分的情况变成2分类问题的话，调用此函数可将3分以下和3分以上(含3分的)变成两类
    if v >= 3:
        return 2
    else:
        return 1

def starTo3Class(v): #如果想把1-5分打分的情况变成3分类（聚类）问题的话，调用此函数可将3分以下,3分和3分以上的变成三类
    if v < 3:
        return 1
    elif v == 3:
        return 2
    elif v <= 5:
        return 3

def splitSentence(text): #中文分句分词，每句变成一个字符串,句内单词之间用空格连接
    #还可以加入对于停用词的考虑
    jieba.load_userdict("jd_Data/user_dict.txt") #加载用户自定义词库
    words = jieba.cut(text)
    splitedSentence = ' '.join(words)
    return splitedSentence

def loadSplitRawData(path='jd_Data/extract_comments4.txt', binary = False, threeClassed = True):
    print('loading JD reviews...')
    columns = ['text','stars']
    df = pd.read_csv(path,header=None,sep='\t',names=columns) 
    df.dropna(axis=0, how='any', inplace=True) #删除有空值的行
    # print ("df from jdData:",df)
    # print (df.shape)
    df['text_tokens'] = df['text'].apply(lambda x: splitSentence(x))
    # print (df['text_tokens'])

    if binary:
        df['binary_stars'] = df['stars'].apply(lambda x: polarize(x))
        df_new = pd.DataFrame(df, columns = ['text_tokens','binary_stars'])
        df_new.rename(columns={ 'text_tokens': 'data', 'binary_stars': 'label'}, inplace=True)
        return df_new
    if threeClassed:
        df['starTo3Class_stars'] = df['stars'].apply(lambda x: starTo3Class(x))
        df_new = pd.DataFrame(df, columns = ['text_tokens','starTo3Class_stars'])
        df_new.rename(columns={ 'text_tokens': 'data', 'starTo3Class_stars': 'label'}, inplace=True)
        return df_new
    else:
        df_new = pd.DataFrame(df, columns = ['text_tokens','stars'])
        df_new.rename(columns={ 'text_tokens': 'data', 'stars': 'label'}, inplace=True)
        return df_new

def load_jdData(path='jd_Data/extract_comments4.txt', binary = False, threeClassed = True):
    if binary == True:
        tfidfFile = 'jdReview_TOP2000tfidf'+'2class'+'.npy'
    if threeClassed == True:
        tfidfFile = 'jdReview_TOP2000tfidf'+'3class'+'.npy'
    data_dir = 'jd_Data/'

    import os
    if os.path.exists(os.path.join(data_dir, tfidfFile)):
        data = np.load(os.path.join(data_dir, tfidfFile)).item()
        x = data['data']
        y = data['label']
        return x,y
 
    df = loadSplitRawData(path, binary, threeClassed)

    data = np.array(df["data"])
    target = np.array(df["label"])

    from sklearn.feature_extraction.text import CountVectorizer
    # Convert a collection of text documents to a matrix of token counts
    # CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数
    # 对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    # Transform a count matrix to a normalized tf or tf-idf representation
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x) # 返回值类型'scipy.sparse.csr.csr_matrix'，下面的todense会将其变为矩阵
    x = x[:].astype(np.float32)
    print(x.dtype, x.size)
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])  # 保证一行的平方和除以D等于1？ D = x.shape[1]
    print('todense succeed')

    p = np.random.permutation(x.shape[0]) #打乱 Randomly permute a sequence, or return a permuted range.
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))

    print (x.shape,y.shape)
    
    np.save(os.path.join(data_dir, tfidfFile), {'data': x, 'label': y}) #Save an array to a binary file in NumPy .npy format.
    return x,y



if __name__ == "__main__":
    load_jdData()