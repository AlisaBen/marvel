from __future__ import print_function
from itertools import count
from collections import defaultdict
from scipy.sparse import csr

import numpy as np
import mxnet as mx
from mxnet import nd
import os
import pandas as pd


def try_gpu(id):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(device_id=id)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def mkdir_if_not_exist(path):
    print('mkdir_if_not_exist:[%s]' % path)
    if not os.path.exists(path):
        os.makedirs(path)


def vectorize_dic(dic, ix=None, p=None):
    """
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature)

    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if (ix == None):
        d = count(0)
        ix = defaultdict(lambda: next(d))

    n = len(list(dic.values())[0])  # num samples
    g = len(list(dic.keys()))  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)

    if (p == None):
        p = len(ix)

    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix

def merge(like_path,finish_path,save_path):
    like_res = pd.read_csv(like_path)
    finish_res = pd.read_csv(finish_path)
    print(like_res.info())
    print(finish_res.info())
    res = pd.concat([like_res,finish_res['finish_probability']],axis=1)
    res.to_csv(save_path,index=False)
    print('merger over')