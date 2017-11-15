# coding: utf-8

import time
import random
import os
import re
import xlwt
import requests
import numpy as np
import xlsxwriter
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sys
import math
reload(sys)
sys.setdefaultencoding('utf8')

os.chdir(u'**********')

guangdabase = pd.read_csv('guangdabase.csv' ,header=None)
guangdabase.head()


def xybasenames(data1):
    data1. colnames =['id', 'UpdateFlag', 'branch', 'ajbh', 'kehu', 'ajlx', 'shfzh', 'shfzh18', 'shebaoID', 'xm', 'pinyin', 'sex',                     'zhiwu', 'zjqkje', 'zjshje', 'zjzxqke', 'zjzxqkerq', 'zjyhlx', 'jdsj', 'dqsj', 'zu', 'ywy', 'states', 'period',                    'yjbl', 'fenpeisj', 'urgent', 'lasttime', 'closetime', 'czy', 'addtime', 'pici', 'inpici', 'shengfen', 'chengshi',                    'remark1', 'remark2', 'remark3', 'lastJzSj', 'kongguan', 'PromisedDate', 'PromisedJe', 'nextStep', 'hint',                     'dingyueTime', 'fabuTime', 'gaNum', 'Ajsx', 'ajInfo', 'kehuAjBh', 'ajStop', 'ajLock', 'yxAj', 'isShare', 'zxxddm',                    'picipizhu']
    return data1

guangdabase =xybasenames(guangdabase[:])

guangdabase.columns = guangdabase.colnames

guangda1 = guangdabase[["ajbh", "shfzh18", "ywy", "zjqkje", "zjshje"]]
guangda1.shape

guangda1["hkzhb"] = guangda1["zjshje"] / guangda1["zjqkje"]
guangda1.describe()

guangda1["bornyear"] = guangda1["shfzh18"].str.slice(6, 10)
guangda1["sex"] = guangda1["shfzh18"].str.get(16)
guangda1["address"] = guangda1["shfzh18"].str.slice(0, 6)
guangda1["shfzhnum"] = guangda1["shfzh18"].str.len()
guangda2 = guangda1[["ajbh", "shfzh18", "bornyear", "shfzhnum", "sex", "address", "zjqkje", "zjshje", "ywy"]]
guangda2.shape

guangda2 = guangda2[guangda2["shfzhnum"] == 18]
guangda2["yearlen"] = guangda2["bornyear"].str.len()
guangda2 = guangda2[guangda2["yearlen"] == 4]
list(set(guangda2["bornyear"]))
guangda2 = guangda2[guangda2["bornyear"] != '\xe7\xac\xac2']
guangda2["bornyear"] = guangda2["bornyear"].astype(int)
guangda2["age"] = 2017 - guangda2["bornyear"]
guangda2["sex"] = guangda2["sex"].astype(int)

guangda2["sex"][guangda2["sex"] % 2 == 0] = 0
guangda2["sex"][guangda2["sex"] != 0] = 1
guangda2["address"] = guangda2["address"].astype(int)
guangda2["hkzhb"] = guangda2["zjshje"] / guangda2["zjqkje"]
guangda2["ywy"] = guangda2["ywy"].str.upper()
guangda3 = guangda2.dropna()
guangda3.shape

guangda = guangda3[["ajbh", "shfzh18", "age", "sex", "address", "zjqkje", "zjshje", "hkzhb", "ywy"]]
guangda = guangda[guangda.zjqkje > 0]
guangda.head()

def maxminscale(normal):
    max1 = np.max(normal)
    min1 = np.min(normal)
    normal = (normal - min1) / (max1 - min1)
    return normal

def datascale(scaledata):
    mean1 = np.mean(scaledata)
    std1 = np.std(scaledata)
    scaledata = (scaledata - mean1) / std1
    return scaledata

guangda4 = guangda

guangda4["age"] = maxminscale(guangda4["age"])
guangda4["address"] = maxminscale(guangda4["address"])
guangda4["zjqkje"] = maxminscale(guangda4["zjqkje"])
guangda4["zjshje"] = maxminscale(guangda4["zjshje"])
guangda4["hkzhb"] = maxminscale(guangda4["hkzhb"])
guangda4.head()

testclust = guangda4[["age", "sex", "address", "zjqkje", "zjshje", "hkzhb"]]
testclust.describe()

a = np.array(range(10))
for j in range(2, 12):
    kmeanss = KMeans(n_clusters=j, init='k-means++', n_init=10, max_iter=300, algorithm='auto').fit(testclust)
    a[j - 2] = kmeanss.inertia_


x = np.array(range(1, 11))

y = a

plt.rc('font', family='SimHei', size=13)
plt.xlabel("聚类个数")
plt.ylabel("均方误差")
plt.plot(x, y)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, algorithm='auto').fit(testclust)
testclust["label"] = kmeans.labels_
kcenters = kmeans.cluster_centers_
kcenters

kmeans.inertia_

def centerDistance(labeldata, centerdata):
    ldata0 = labeldata[labeldata.label == 0]
    ldata0 = ldata0.iloc[:, 0:6]
    ldata00 = np.array(ldata0)

    ldata1 = labeldata[labeldata.label == 1]
    ldata1 = ldata1.iloc[:, 0:6]
    ldata11 = np.array(ldata1)

    ldata2 = labeldata[labeldata.label == 2]
    ldata2 = ldata2.iloc[:, 0:6]
    ldata22 = np.array(ldata2)

    ldata3 = labeldata[labeldata.label == 3]
    ldata3 = ldata3.iloc[:, 0:6]
    ldata33 = np.array(ldata3)

    ldata4 = labeldata[labeldata.label == 4]
    ldata4 = ldata4.iloc[:, 0:6]
    ldata44 = np.array(ldata4)

    test0 = ldata00 - kcenters[0]
    test0 = test0 * test0
    test0 = test0.sum(axis=1)
    test0 = np.sqrt(test0)

    test1 = ldata11 - kcenters[1]
    test1 = test1 * test1
    test1 = test1.sum(axis=1)
    test1 = np.sqrt(test1)

    test2 = ldata22 - kcenters[2]
    test2 = test2 * test2
    test2 = test2.sum(axis=1)
    test2 = np.sqrt(test2)

    test3 = ldata33 - kcenters[3]
    test3 = test3 * test3
    test3 = test3.sum(axis=1)
    test3 = np.sqrt(test3)

    test4 = ldata44 - kcenters[4]
    test4 = test4 * test4
    test4 = test4.sum(axis=1)
    test4 = np.sqrt(test4)

    test = np.concatenate((test0, test1, test2, test3, test4))

    return test

ttttt = centerDistance(testclust, kcenters)

testclust["ddd"] = ttttt
testclust.head()

guangda["label"] = testclust["label"]
guangda["ddd"] = testclust["ddd"]
testclust["ywy"] = guangda["ywy"]
guangda["nnn"] = 1
guangda.head()

data1 = guangda.groupby("label").sum().reset_index()
data1 = data1[["label", "nnn"]]

data2 = guangda.groupby(["label", "ywy"]).sum().reset_index()
data2 = data2.sort(["label", "zjshje"], ascending=False)
data2 = data2[["label", "ywy", "zjshje", "nnn"]]

fpresult = pd.merge(data2, data1, on="label", how="left")
fpresult["fpb"] = fpresult["nnn_x"] / fpresult["nnn_y"]
fpresult = fpresult[["label", "ywy", "zjshje", "nnn_x", "fpb"]]

guangdaresult = pd.merge(guangda, data3, on=["label", "ywy"], how="left")
rankdata = guangdaresult.sort(["label", "ddd"])
rankdata[rankdata.label == 4]

label4 = fpresult[fpresult.label == 4].shape[0]
fpdata4 = fpresult[fpresult.label == 4]
rankdata4 = rankdata[rankdata.label == 4]
sumnum = 0
for i in range(label4):
    rank4 = int(fpdata4["nnn_x"][i])
    sumnum = sumnum + rank4
    sumnum1 = sumnum - rank4
    if sumnum < rank4 + 1:
        rankdata4["ywy"][0:sumnum] = fpdata4["ywy"][i]
    else:
        rankdata4["ywy"][sumnum1:sumnum] = fpdata4["ywy"][i]

label3 = fpresult[fpresult.label == 3].shape[0]
fpdata3 = fpresult[fpresult.label == 3]
rankdata3 = rankdata[rankdata.label == 3]
sumnum = 0
for i in range(label3):
    rank3 = int(fpdata3["nnn_x"][i])
    sumnum = sumnum + rank3
    sumnum1 = sumnum - rank3
    if sumnum < rank3 + 1:
        rankdata3["ywy"][0:sumnum] = fpdata3["ywy"][i]
    else:
        rankdata3["ywy"][sumnum1:sumnum] = fpdata3["ywy"][i]

label2 = fpresult[fpresult.label == 2].shape[0]
fpdata2 = fpresult[fpresult.label == 2]
rankdata2 = rankdata[rankdata.label == 2]
sumnum = 0
for i in range(label2):
    rank2 = int(fpdata2["nnn_x"][i])
    sumnum = sumnum + rank2
    sumnum1 = sumnum - rank2
    if sumnum < rank2 + 1:
        rankdata2["ywy"][0:sumnum] = fpdata2["ywy"][i]
    else:
        rankdata2["ywy"][sumnum1:sumnum] = fpdata2["ywy"][i]

label1 = fpresult[fpresult.label == 1].shape[0]
fpdata1 = fpresult[fpresult.label == 1]
rankdata1 = rankdata[rankdata.label == 1]
sumnum = 0
for i in range(label1):
    rank1 = int(fpdata1["nnn_x"][i])
    sumnum = sumnum + rank1
    sumnum1 = sumnum - rank1
    if sumnum < rank1 + 1:
        rankdata1["ywy"][0:sumnum] = fpdata1["ywy"][i]
    else:
        rankdata1["ywy"][sumnum1:sumnum] = fpdata1["ywy"][i]

label0 = fpresult[fpresult.label == 0].shape[0]
fpdata0 = fpresult[fpresult.label == 0]
rankdata0 = rankdata[rankdata.label == 0]
sumnum = 0
sumnum1 = 0
for i in range(label0):
    rank0 = int(fpdata0["nnn_x"][i])
    sumnum = sumnum + rank0
    sumnum1 = sumnum - rank0
    if sumnum < rank0 + 1:
        rankdata0["ywy"][0:sumnum] = fpdata0["ywy"][i]
    else:
        rankdata0["ywy"][sumnum1:sumnum] = fpdata0["ywy"][i]

resultrankdata = pd.concat(rankdata0, rankdata1, rankdata2, rankdata3, rankdata4)
resultrankdata