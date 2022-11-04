#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:17:08 2022

@author: lidor
"""
import numpy as np
from ccg import *
import create_feat_tsc as tsc
import test_sort_shank9_4 as MC
from prepare_features_test3 import prepare_features_test3

def getX(item):
    x = X1[:,item].float().unsqueeze(0)
    return x

def getY(item):
    y = str(int(Y[item]))
    return y

# Loading files 
# the res, clu mspk, and sspke are created seperatly 
# in different part of the code (by Youcef) 
path1  = '/home/lidor/data/DBC/export_before_classifiers/23_414_Kilosort_06_29_2022_08_32_20.'  
res    = np.load(path1+'res.1.npy')
clu    = np.load(path1+'clu.1.npy')
mspk   = np.load(path1+'mspk.1.npy')
sspk   = np.load(path1+'sspk.1.npy')


# make new featurs for the second part of the AI algo
sample_rate    = 20000

U1      = 9
U2      = 579
clu1    = clu[(clu==U1) | (clu==U2)]
res1    =res[(clu==U1) | (clu==U2)]


cc                = compCCG(res1,clu1,FS=sample_rate,window_size=0.04)[0]
nspk_vec           = tsc.compute_Nvec(clu1)
cluster_ids        = np.unique(clu1)
time_mat           = tsc.compute_timeMat(clu1,res1,cluster_ids)

ind            = np.where((np.unique(clu)==U1)|(np.unique(clu)==U2))[0]
m1             =mspk[:,:,ind]
s1             =sspk[:,:,ind]

x = prepare_features_test3(0, 1, clu1, m1, s1, cc, time_mat, cluster_ids,0)
print(np.argmax( MC.predict2(x)[0]),np.max( MC.predict2(x)[0]))

#-----------------------------------------------
# cluster_ids        = np.unique(clu)
# cc                 = compCCG(res,clu,FS=sample_rate,window_size=0.04)[0]
# nspk_vec           = tsc.compute_Nvec(clu)
# cluster_ids        = np.unique(clu)
# time_mat           = tsc.compute_timeMat(clu,res,cluster_ids)
# i =0.95
# feat_mat = np.empty((1, 5, 128))
# n = len(cluster_ids)
# for m in range(n):
#     x = prepare_features_test3(4, m, clu, mspk, sspk, cc, time_mat, cluster_ids,1)
#     feat_mat = np.concatenate((feat_mat, x), 0)
# if len(feat_mat) > 1:
#     feat_mat = np.delete(feat_mat, 0, axis=0)
#     a   = MC.predict2(feat_mat)
#     idx = np.where(a[:, 1] > i)
#     idx = idx[0]