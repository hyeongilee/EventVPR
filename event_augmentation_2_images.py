#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from IPython.display import HTML
import zipfile

from scipy.spatial import distance
import time


# In[2]:


################################no_grid####################################
env_d = "clean"
date_d = "2020_07_04_11_08_58"
env_q = "rain"
date_q = "2020_07_30_17_51_53" #0704/ 110858 111638 112402 // 0730/ 175153 181705 182029 // 0730
net_name = "resnet"
code = "14"                         ####################code change####################
resolution = (224, 224)

def viewpoint_crop(img, a, b):

    data = list()

    x_i = random.randint(0,20)
    y_i = random.randint(a,b)
    x_f = x_i + 300
    y_f = y_i + 200

    for item in img:
        if (item[0] >= x_i and item[0] < x_f) and (item[1] >= y_i and item[1] < y_f):
            item[0] -= x_i
            item[1] -= y_i
            data.append(item)
    return data

def augmentation(num_d, num_q):
    E_d = np.load("../dataset/KH/"+env_d+"/raw_numpy_100ms/"+date_d+"/"+str(num_d)+".npy")
    E_q = np.load("../dataset/KH/"+env_q+"/raw_numpy_100ms/"+date_q+"/"+str(num_q)+".npy")

    E_d = E_d.tolist()
    E_q = E_q.tolist()
    N_target = min(len(E_d), len(E_q))
    N_base = max(len(E_d), len(E_q))
    a_f = N_target/N_base
    print(N_target, N_base, a_f)
    if(a_f < 0.01 or N_target > 300000 or N_base < 10000):
        print('reject')
        return []
    a_f_d = 0
    a_f_q = 0
    if(len(E_d)>=len(E_q)):
        a_f_d = a_f
        a_f_q = 1
    else:
        a_f_d = 1
        a_f_q = a_f
    D_d = dict()
    data_d = list()
    cnt_d = 0
    for item in E_d:
        x = item[0]
        y = item[1]
        p = item[2]
        t = item[3]
        if not ((x, y) in D_d):
            D_d[(x, y)] = list()
            D_d[(x, y)].append(item)
        elif D_d[(x, y)][0][2] == p:
            D_d[(x, y)].append(item)
        elif D_d[(x, y)][0][2] != p:
            first = D_d[(x, y)][0][3]
            last = D_d[(x, y)][-1][3]
            Ne = len(D_d[(x, y)])
            Ne_a = round(Ne * a_f)
            if Ne == 1:
                r = random.randint(0,9)
                if r < int(10*a_f_d):
                    data_d.append(D_d[(x, y)][0])
                    D_d[(x, y)] = list()
                    D_d[(x, y)].append(item)
                continue
            elif first == last:
                for j in range(Ne_a):
                    data_d.append(D_d[(x, y)][0])
                    #cnt+=1
                D_d[(x, y)] = list()
                D_d[(x, y)].append(item)
                continue
            if Ne_a == 0:
                D_d[(x, y)] = list()
                D_d[(x, y)].append(item)
                continue
            #print(Ne_a)
            #print(last, first)
            times = np.arange(first, last, float((last-first)/Ne_a))#+0.0001)
            #print(times)
            times = times.tolist()
            #print(len(times),Ne_a,Ne)
            for i in range(len(times)):
                e = [x, y, D_d[(x, y)][0][2], round(times[i])]
                data_d.append(e)
                #cnt += 1
            D_d[(x, y)] = list()
            D_d[(x, y)].append(item)

    for key in D_d:
        first = D_d[key][0][3]
        last = D_d[key][-1][3]
        Ne = len(D_d[key])
        Ne_a = round(Ne * a_f_d)
        if Ne == 1:
            r = random.randint(0,99)
            if r < int(100*a_f_d):
                data_d.append(D_d[key][0])
            continue
        if first == last and Ne != 1:
            for j in range(Ne_a):
                data_d.append(D_d[key][0])
                #cnt+=1
            continue
        if Ne_a == 0:
            continue
        times = np.arange(first, last, float((last-first)/Ne_a))#+0.0001)
        times = times.tolist()
        for i in range(len(times)):
            e = [key[0], key[1], D_d[key][0][2], round(times[i])]
            data_d.append(e)
            #cnt += 1
    data_d = sorted(data_d, key = lambda x : x[3])

    #/////////////

    D_q = dict()
    data_q = list()
    cnt_q = 0
    for item in E_q:
        x = item[0]
        y = item[1]
        p = item[2]
        t = item[3]
        if not ((x, y) in D_q):
            D_q[(x, y)] = list()
            D_q[(x, y)].append(item)
        elif D_q[(x, y)][0][2] == p:
            D_q[(x, y)].append(item)
        elif D_q[(x, y)][0][2] != p:
            first = D_q[(x, y)][0][3]
            last = D_q[(x, y)][-1][3]
            Ne = len(D_q[(x, y)])
            Ne_a = round(Ne * a_f)
            if Ne == 1:
                r = random.randint(0,9)
                if r < int(10*a_f_q):
                    data_q.append(D_q[(x, y)][0])
                    D_q[(x, y)] = list()
                    D_q[(x, y)].append(item)
                continue
            elif first == last:
                for j in range(Ne_a):
                    data_q.append(D_q[(x, y)][0])
                    #cnt+=1
                D_q[(x, y)] = list()
                D_q[(x, y)].append(item)
                continue
            if Ne_a == 0:
                D_q[(x, y)] = list()
                D_q[(x, y)].append(item)
                continue
            #print(Ne_a)
            #print(last, first)
            times = np.arange(first, last, float((last-first)/Ne_a))#+0.0001)
            #print(times)
            times = times.tolist()
            #print(len(times),Ne_a,Ne)
            for i in range(len(times)):
                e = [x, y, D_q[(x, y)][0][2], round(times[i])]
                data_q.append(e)
                #cnt += 1
            D_q[(x, y)] = list()
            D_q[(x, y)].append(item)

    for key in D_q:
        first = D_q[key][0][3]
        last = D_q[key][-1][3]
        Ne = len(D_q[key])
        Ne_a = round(Ne * a_f_q)
        if Ne == 1:
            r = random.randint(0,99)
            if r < int(100*a_f_q):
                data_q.append(D_q[key][0])
            continue
        if first == last and Ne != 1:
            for j in range(Ne_a):
                data_q.append(D_q[key][0])
                #cnt+=1
            continue
        if Ne_a == 0:
            continue
        times = np.arange(first, last, float((last-first)/Ne_a))#+0.0001)
        times = times.tolist()
        for i in range(len(times)):
            e = [key[0], key[1], D_q[key][0][2], round(times[i])]
            data_q.append(e)
            #cnt += 1
    data_q = sorted(data_q, key = lambda x : x[3])

    data_d = viewpoint_crop(data_d, 0, 5)   #0,5
    data_q = viewpoint_crop(data_q, 26, 31) #26,31

    x_res = resolution[0]
    y_res = resolution[1]
    for i in range(len(data_d)):
        data_d[i][0] = int(data_d[i][0]*(x_res/300.0))
        data_d[i][2] = int(data_d[i][1]*(y_res/200.0))
        data_d[i][2] = 0
    for i in range(len(data_q)):
        data_q[i][0] = int(data_q[i][0]*(x_res/300.0))
        data_q[i][2] = int(data_q[i][1]*(y_res/200.0))
        data_q[i][2] = 1
    data = data_d
    data.extend(data_q)
    data = np.array(data)
    print(data.shape)
    return data


# In[3]:
ms = 100
database_utm_file = open("../dataset/KH/"+env_d+"/raw_numpy_"+str(ms)+"ms/"+date_d+"_location.txt",'r')
query_utm_file = open("../dataset/KH/"+env_q+"/raw_numpy_"+str(ms)+"ms/"+date_q+"_location.txt",'r')

output_location = "/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/aug_"+net_name+"_"+str(resolution[0])+"x"+str(resolution[1])+"/"

ground_file = open(output_location+"train.txt",'a')

threshold = 3 #m
threshold2 = 50 #m

n_true = 0
n_false = 0
D_utm = list()
Q_utm = list()
S = set()

while True:
    line = database_utm_file.readline()
    if not line:
        break
    line_list = line.split()
    item = [int(line_list[0]),float(line_list[1]),float(line_list[2])]
    D_utm.append(item)
database_utm_file.close()

while True:
    line = query_utm_file.readline()
    if not line:
        break
    line_list = line.split()
    item = [int(line_list[0]),float(line_list[1]),float(line_list[2])]
    Q_utm.append(item)
query_utm_file.close()


while n_true < 5000 or n_false < 5000:
    d_num = random.randint(0,len(D_utm)-1)
    q_num = random.randint(0,len(Q_utm)-1)
    if (d_num, q_num) in S:
        continue

    utm_d = (D_utm[d_num][1], D_utm[d_num][2])
    utm_q = (Q_utm[q_num][1], Q_utm[q_num][2])

    d = distance.euclidean(utm_d, utm_q)

    if d > threshold and d < threshold2:
        S.add((d_num, q_num))
        continue
    elif d <= threshold:
        result = augmentation(d_num, q_num)
        if result == []:
            S.add((d_num, q_num))
            continue
        S.add((d_num, q_num))
        ground_file.write(code+'0'*(4-len(str(d_num)))+str(d_num)+'0'+'0'*(4-len(str(q_num)))+str(q_num)+"\t1\n")
        n_true += 1
    elif d >= threshold2:
        if n_false >= n_true:
            continue
        result = augmentation(d_num, q_num)
        if result == []:
            S.add((d_num, q_num))
            continue
        S.add((d_num, q_num))
        ground_file.write(code+'0'*(4-len(str(d_num)))+str(d_num)+'0'+'0'*(4-len(str(q_num)))+str(q_num)+"\t0\n")
        n_false += 1

    





 
    
        
    
    np.save(output_location+code+'0'*(4-len(str(d_num)))+str(d_num)+'0'+'0'*(4-len(str(q_num)))+str(q_num)+".npy",result)

# In[ ]:




