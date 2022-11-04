
import numpy as np
from prepare_features_test3 import trim_spk_4ch,get_time_fet
from fastai.text.all import *

def prepare_features_pair(i, j, clu, mean_spk, std_spk, cc, time_mat, u_clu):
    idx1 = np.where(clu == u_clu[i])
    idx2 = np.where(clu == u_clu[j])

    mean_spk1 = mean_spk[:, :, i]
    [mean_spk1, ind] = trim_spk_4ch(mean_spk1)
    mean_spk1 = mean_spk1.flatten()
    mean_spk2 = (mean_spk[ind, :, j]).flatten()
    max1 = max(max(abs(mean_spk1)), max(abs(mean_spk2)))

    std_spk1 = (std_spk[ind, :, i]).flatten()
    std_spk2 = (std_spk[ind, :, j]).flatten()

    if np.sum(cc[:, i, i]) == 0:
        acc1 = cc[:, i, i]
    else:
        acc1 = cc[:, i, i] / np.max(cc[:, i, i])
    acc1 = acc1[20:]

    if np.sum(cc[:, j, j]) == 0:
        acc2 = cc[:, j, j]
    else:
        acc2 = cc[:, j, j] / np.max(cc[:, j, j])
    acc2 = acc2[20:]

    if np.sum(cc[:, i, j]) == 0:
        ccgtag = cc[:, i, j]
    else:
        ccgtag = cc[:, i, j] / np.max(cc[:, i, j])

    i1 = len(idx1[0])
    i2 = len(idx2[0])
    n = np.minimum(i1, i2) / np.maximum(i1, i2)
    t = get_time_fet(time_mat, i, j)

    last_row = np.concatenate((acc1.T, acc2.T, ccgtag.T, np.array([n]), t.flatten()))
    x = (np.concatenate((mean_spk1, mean_spk2, std_spk1, std_spk2))) / max1
    x = np.concatenate((x, last_row))
    x = np.reshape(x, (5, 128, 1))
    x = np.moveaxis(x, -1, 0)
    return x

def predict2(feat_mat):
    clf = load_learner('incep_25_10.pkl')
    feat_mat = tensor(feat_mat)
    feat_mat[:, 4, 0:83] += -0.5
    dl = clf.dls.test_dl(feat_mat)
    preds = clf.get_preds(dl=dl)
    a = preds[0].numpy()
    return a

def load_data(filebase, shank_num):
    f = open(filebase+'/info')
    info = f.read()
    lst_info = info.split('\n')
    session_name = lst_info[0]

    clu = np.load(filebase + '/' + session_name + '.clu.' + shank_num + '.npy')
    res = np.load(filebase + '/' + session_name + '.res.' + shank_num + '.npy')
    cc = np.load(filebase + '/' + session_name + '.cc.' + shank_num + '.npy')
    mspk = np.load(filebase + '/' + session_name + '.mspk.' + shank_num + '.npy')
    sspk = np.load(filebase + '/' + session_name + '.sspk.' + shank_num + '.npy')
    nspk_vec = np.load(filebase + '/' + session_name + '.nspk_vec.' + shank_num + '.npy')
    time_mat = np.load(filebase + '/' + session_name + '.timeMat.' + shank_num + '.npy')
    nspk_vec = np.squeeze(nspk_vec)
    return clu, res, cc, mspk, sspk, nspk_vec, time_mat

def getX(file):
    data = np.load(file)
    x = data[:, :128]
    x[4,0:83] += -0.5
    return torch.FloatTensor(x)


def getY(file):
    data = np.load(file)
    y = data[0, -1]
    y = str(int(y))
    return y


import sys
k = len(sys.argv)-1
filebase = ''
for i in range(k):
    if i == 0:
        filebase += sys.argv[i+1]
    else:
        filebase = filebase + ' ' + sys.argv[i+1]
# filebase = '/home/tali/Documents/mP79_15/npy_files'
f = open(filebase + '/info')
info = f.read()
lst_info = info.split('\n')
shank_num = str(lst_info[1])
f.close()
clu, res, cc, mean_spk, std_spk, nspk_vec, time_mat = load_data(filebase, shank_num)
u_clu = np.unique(clu)
features = prepare_features_pair(0, 1, clu, mean_spk, std_spk, cc, time_mat, u_clu)
a = predict2(features)
f2 = open(filebase + '/prediction','w')
f2.write(str(a[0,1]))
f2.close()
