# -*- coding: utf-8 -*-

'''
读入数据和模型，展示距离每个聚类中心最近的若干条数据
'''
# 读取原始数据和对应的index
# 读取处理后的数据和index
# 从文件读取模型
# 得到某一层训练后的权重和所有样本在这一层的输入
# 对每个样本，计算其到若干个中心的距离，
# 对每一个聚类中心，找到距离其最近的若干条数据，以及他们真实对应的标签，打印出来
from DEC import DEC
import numpy as np
import pandas as pd
from keras.models import Model
from keras.initializers import VarianceScaling
from datasets import load_data
import heapq # heap queue algorithm

## 京东5星评论三分类
datasetKey = 'jd' # 数据集关键字
weightsFile = 'results/jd_3class/DEC_model_final.h5' # 训练好的模型权重
rawDataPath='jd_Data/extract_comments4.txt'

## 京东评论 with label
# datasetKey = 'jdReviewWithLabel' # 数据集关键字
# weightsFile = 'results/jd_reviewWithLabel/DEC_model_final.h5' # 训练好的模型权重
# rawDataPath='jd_Data/negatives_withlabel.txt'

# 读入原始数据
columns = ['text','label']
df = pd.read_csv(rawDataPath,header=None,sep='\t',names=columns) 
df.dropna(axis=0, how='any', inplace=True) #删除有空值的行

# 读入训练数据
x, y = load_data(datasetKey)
n_clusters = len(np.unique(y))

# 建模并load训练好的权重
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init) #dims参数规定了stacked autoencoder的结构
dec.load_weights(weightsFile)
dec.model.summary()

# 为了获得中间层的输出，一种简单的方法是创建一个新的Model，使得它的输出是你想要的那个输出
# 获得 encoder_3 层的输出，它相当于每个样本经过编码后的结果，相当于论文中的z
layer_name = 'encoder_3'
intermediate_layer_model = Model(input=dec.model.input,
                                 output=dec.model.get_layer(layer_name).output)
encodedData = intermediate_layer_model.predict(x)
# print (encodedData.shape)  # (50000,10)


# 读取clustering层的权重, 它相当于是若干个聚类中心的表示miu
# 通过依次计算各个样本的z与各个聚类中心miu的距离，可以得到距离每个聚类中心最近的若干个样本点
clusteringLayer = dec.model.get_layer(name = 'clustering')
clusterCenters = clusteringLayer.get_weights()   #获取该层的参数,即若干个聚类中心
# print (len(clusterCenters)) # 1
# print (len(clusterCenters[0])) # 3
# print (clusterCenters) 

## 找到距离各个聚类中心最近的若干个样本点对应的index
distence2centers = []
for center_i in clusterCenters[0]:
    distences = []
    for data_i in encodedData:
        distences.append(np.linalg.norm(center_i - data_i))
    distence2centers.append(distences)

nearestNUM = 20
nearestINDEX = []
for i in range(len(distence2centers)):
    distences = distence2centers[i]
    nearestIndexList = map(distences.index, heapq.nsmallest(nearestNUM, distences))  # 先找到最小的距离，再找到最小的距离对应的索引
    nearestINDEX.append(nearestIndexList)


## 对聚好的每一类中离中心最近的若干个点，打印原始样本信息和真实的标签
pd.set_option('max_colwidth',100) #避免文本显示不全的问题
for j in range(len(nearestINDEX)):
    print ("\ncluster",j,":")
    print (df.loc[nearestINDEX[j],:])