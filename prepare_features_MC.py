import numpy as np
from scipy.spatial import distance


def get_time_fet(time_mat, i, j):
    v1 = time_mat[i, :]
    v2 = time_mat[j, :]
    m1 = np.mean(v1)
    m2 = np.mean(v2)
    threshold1 = m1 * 0.1
    threshold2 = m2 * 0.1
    v1[np.where(v1 <= threshold1)] = 0
    v2[np.where(v2 <= threshold2)] = 0
    v3 = np.zeros((44, 1))

    for k in range(44):
        if v1[k] == 0:
            if v2[k] != 0:
                v3[k] = 0.5
            else:
                v3[k] = 0
        elif v2[k] == 0:
            v3[k] = -0.5
        else:
            v3[k] = 1
    return v3


def prepare_features_MC(i, j, clu, mean_spk, std_spk, cc, time_mat, u_clu, k):
    if (sum(cc[:, i, j]) < 400) & (k == 0):
        return np.ones((1, 5, 128))

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

    # for xgboost
    # s = x.shape
    # x = np.moveaxis(x, -1, 0)
    # x = np.reshape(x, (s[2], -1), 'F')
    return x


def trim_spk_4ch(mean_spk):
    n_channels = np.size(mean_spk, 0)
  #  max_idx = np.argmax(mean_spk[:, 15], 0)
    if n_channels < 4:
        new_mspk = mean_spk
        channels_idx = np.arange(0, n_channels)
        channels_idx = channels_idx.T
    else:
        M1 = np.amax(mean_spk, axis=1)
        M2 = np.amin(mean_spk, axis=1)
        I = np.argsort(np.abs(M1 - M2))
        channels_idx = I[-4:]
        channels_idx = np.flip(channels_idx)
        new_mspk = mean_spk[channels_idx.T, :]

    return new_mspk, channels_idx


# filebase = '/home/tali/matlab/Database/npy_mP31_07_test'
# name = 'mP31_07'
# shank_num = '1'
# clu = np.load(filebase + '/' + name + '.clu.' + shank_num + '.npy')
# res = np.load(filebase + '/' + name + '.res.' + shank_num + '.npy')
# cc = np.load(filebase + '/' + name + '.cc.' + shank_num + '.npy')
# time_mat = np.load(filebase + '/' + name + '.time_mat.' + shank_num + '.npy')
# mspk = np.load(filebase + '/' + name + '.mspk.' + shank_num + '.npy')
# sspk = np.load(filebase + '/' + name + '.sspk.' + shank_num + '.npy')
# # nspk_vec = np.load(filebase + '/' + name + '.nspk_vec.' + shank_num + '.npy')
# # nspk_vec = np.squeeze(nspk_vec)
# u_clu = np.unique(clu)
# i = 0
# j = 4
# x = prepare_features_test(i, j, clu, mspk, sspk, cc, time_mat, u_clu)
# x2 = np.load('/home/tali/matlab/Database/npy_mP31_07_test/mP31_07.x.1-5.npy')
# x2 = np.reshape(x2, (1, -1), 'F')

