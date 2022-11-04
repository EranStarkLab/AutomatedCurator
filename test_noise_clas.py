#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:52:43 2022

@author: lidor
"""
import numpy as np
from ccg import *
import create_feat_tsc as tsc
import test_sort_shank9_4 as MC

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

# Creating features for the from the res, clu mspk ans sspk for the noise neuronal classifaire 
cc       = compCCG(res,clu,FS=20000,window_size=0.062)
cc       = cc[0][1:-1,:,:] 
X        = tsc.creatFeat_all(mspk,sspk,cc)
# predict which cluster is noise or neuronal
pred     = tsc.predict_tsc(X)
cleanClu = tsc.tsc(pred,clu)
ind,Z    = tsc.get_cleanClu_idx(clu,cleanClu)