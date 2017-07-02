# -*- coding: utf-8 -*-
#
# Author: Wan Xiaolin <wanxl13@lzu.edu.cn>

"""Neural Computer Module

"""

import collections
import functools
import itertools
import math

import numpy
import tensorflow
from tensorflow import (concat, expand_dims, reverse, scatter_nd, shape,
                        split, stack, squeeze, transpose)
from tensorflow import multiply, sigmoid, tanh, norm, exp
from tensorflow import reduce_sum as rsum, cumprod
from tensorflow.python.ops.nn_ops import top_k, softplus

__all__ = ['tensorslice', 'intfcparse', 'contentaddress', 'allocmem',
           'tpmemrecall', 'readweight', 'writeweight', 'memread', 'memwrite',
           'NeuralCPU']

DTYPE = 'float16'


def get_rand_weights(nrows, ncols, name=None):
    w = numpy.random.randn(nrows, ncols)
    return tensorflow.Variable(w.astype(DTYPE), name=name)


def get_xavier_weights(nrows, ncols, name=None):
    high = math.sqrt(6) / math.sqrt(nrows + ncols)
    low = -high
    w = numpy.random.uniform(low=4 * low, high=high, size=(nrows, ncols))
    return tensorflow.Variable(w.astype(DTYPE), name=name)


def get_zero_weights(nrows, ncols, name=None):
    w = numpy.zeros(shape=(nrows, ncols))
    return tensorflow.Variable(w.astype(DTYPE), name=name)


def oneplus(x):
    return 1 + softplus(x)


def softmax(x, axis):
    return exp(x) / rsum(exp(x), axis=axis, keep_dims=True)


def tensorslice(tensor, n_slices):
    """Tensor slice

    Parameters
    ----------
    tensor :
    n_slices : `[s1_begin, s1]`

    Returns
    -------

    """
    slice_idx = [slice(idx[0], idx[1])
                 for idx in zip(n_slices[0:-1], n_slices[1:])]
    return [tensor[s] for s in slice_idx]


def intfcparse(xi, word_size, n_rh):
    """Interface parse.

    Parameters
    ----------
    xi : `[batch_size, (word_size*n_rh + word_size*3 + n_rh*5 + 3)]`
    word_size : i.e. `config.word_size`.
    n_rh : number of read heads, i.e. `config.n_readheads`

    Returns
    -------
    key_r : a R-list, each element has size of `[batch_size, word_sz]`.
    beta_r : a R-list, each element has size of `[batch_size, 1]`.
    key_w : `[batch_size, word_sz]`.
    beta_w : scalar.
    e : `[batch_size, word_size]`.
    v : `[batch_size, word_size]`.
    fg : a R-list, each element has size of `[batch_size, 1]`.
    g_a : a R-list, each element has size of `[batch_size, 1]`.
    g_w : a R-list, each element has size of `[batch_size, 1]`.
    pi : a R-list, each element has size of `[batch_sz, 3]`.
    """
    rh_nsplits_1 = list(itertools.repeat(word_size, times=n_rh))
    rh_nsplits_2 = list(itertools.repeat(1, times=n_rh))
    rh_nsplits_3 = list(itertools.repeat(3, times=n_rh))
    items_size = [n_rh*word_size, n_rh, word_size, 1, word_size, word_size,
                  n_rh, 1, 1, n_rh*3]
    items_name = ['key_r', 'beta_r', 'key_w', 'beta_w', 'e', 'v', 'fg', 'g_a',
                  'g_w', 'pi']
    intfc = dict(zip(items_name, split(xi, items_size, axis=1)))
    intfc['key_r'] = split(intfc['key_r'], rh_nsplits_1, axis=1)
    intfc['beta_r'] = split(oneplus(intfc['beta_r']), rh_nsplits_2, axis=1)
    intfc['key_w'] = intfc['key_w']
    intfc['beta_w'] = oneplus(intfc['beta_w'])
    intfc['e'] = sigmoid(intfc['e'])
    intfc['v'] = intfc['v']
    intfc['fg'] = split(sigmoid(intfc['fg']), rh_nsplits_2, axis=1)
    intfc['g_a'] = sigmoid(intfc['g_a'])
    intfc['g_w'] = sigmoid(intfc['g_w'])
    intfc['pi'] = [softmax(x, axis=1)
                   for x in split(intfc['pi'], rh_nsplits_3, axis=1)]
    return intfc


def contentaddress(key, mem, beta):
    """Content address

    Parameters
    ----------
    key : `[batch_size, word_size]`
    mem : `[batch_size, mem_size, word_size]`
    beta : `[batch_size, 1]`

    Returns
    -------
    `[batch_size, mem_size]`
    """
    key_ = expand_dims(key, axis=2)
    norm_mem = norm(mem, axis=2)
    norm_key = norm(key, axis=1, keep_dims=True)
    cos_sim = squeeze(mem @ key_) / (norm_key * norm_mem)
    return softmax(cos_sim * beta, axis=1)


def allocmem(u_tm1, ww_tm1, wr_tm1_ls, fg_ls):
    """Allocate Memory.

    Parameters
    ----------
    u_tm1 : `[batch_size, mem_size]`.
    ww_tm1 : `[batch_size, mem_size]`.
    wr_tm1_ls : a list of R read weights. each element in the list has size of:
        `[batch_size, mem_size]`.
    fg_ls : a list of R free gates. each element in the list has size of:
        `[batch_size, 1]`.

    Returns
    -------
    u : `[batch_size, mem_size]`.
    alloc_vec : `[batch_size, mem_size]`
    """
    mem_size = shape(u_tm1)[1]
    retention = functools.reduce(
        multiply, [1 - fg * wr_tm1 for fg, wr_tm1 in zip(fg_ls, wr_tm1_ls)])
    u = (u_tm1 + ww_tm1 - u_tm1 * ww_tm1) * retention
    asd_u, asd_u_idx = top_k(u, k=mem_size)

    idx = reverse(asd_u_idx, axis=[1])
    prod_phi = cumprod(reverse(asd_u, axis=[1]), axis=1, exclusive=True)
    alloc_vec = (1 - u) * prod_phi
    return alloc_vec, u


def tpmemrecall(tpmem_tm1, p_tm1, ww, wr_tm1_ls):
    """Temporal memory recall.

    Parameters
    ----------
    tpmem_tm1 : Temporal memory, `[batch_size, mem_size, mem_size]`.
    p_tm1 : `[batch_size, mem_size]`.
    ww : write weights, `[batch_size, mem_size]`
    wr_tm1_ls : read weights list, each element in the list has size of:
        `[batch_size, mem_size]`

    Returns
    -------
    """
    p = (1 - rsum(ww)) * p_tm1 + ww
    tpmem = ((1 - expand_dims(ww, axis=2) - expand_dims(ww, axis=1)) *
             tpmem_tm1 + expand_dims(p_tm1, axis=2) @ expand_dims(ww, axis=1))

    tp_tpmem = transpose(tpmem, perm=[0, 2, 1])
    fseq = [squeeze(expand_dims(x, axis=1) @ tpmem) for x in wr_tm1_ls]
    bseq = [squeeze(expand_dims(x, axis=1) @ tp_tpmem) for x in wr_tm1_ls]
    return tpmem, p, fseq, bseq


def readweight(fseq_ls, bseq_ls, content_r_ls, pi_ls):
    """Return read weighting list `wr`.

    Parameters
    ----------
    fseq_ls : `[batch_size, mem_size]`.
    bseq_ls : `[batch_size, mem_size]`.
    content_r_ls : `[batch_size, mem_size]`.
    pi_ls : `[batch_size, 3]`.

    Returns
    -------
    """
    wr_ls = list()
    for fseq, bseq, cont_r, pi in zip(fseq_ls, bseq_ls, content_r_ls, pi_ls):
        v = expand_dims(pi, axis=1) * stack([bseq, cont_r, bseq], axis=2)
        wr_ls.append(rsum(v, axis=2))
    return wr_ls


def writeweight(alloc, content_w, g_w, g_a):
    """Return write weighting `ww`

    Parameters
    ----------
    alloc : `[batch_size, mem_size]`.
    content_w : `[batch_size, mem_size]`.
    g_w : `[batch_size, 1]`.
    g_a : `[batch_size, 1]`.

    Returns
    -------
    ww : `[batch_size, mem_size]`.
    """
    return g_w * (g_a * alloc + (1 - g_a) * content_w)


def memread(mem, wr_ls):
    """Memory read head

    Parameters
    ----------
    mem : `[batch_size, mem_size, word_size]`.
    wr_ls : a R-list, each element has size of `[batch_size, mem_size]`.

    Returns
    -------
    r_ls : `[batch_size, word_size]`
    """
    return [squeeze(expand_dims(wr, axis=1) @ mem) for wr in wr_ls]


def memwrite(mem_tm1, ww, erase, write_vec):
    """Memory write head.

    Parameters
    ----------
    mem_tm1 : `[batch_size, mem_size, word_size]`.
    ww : `[batch_size, mem_size]`.
    erase : `[batch_size, word_size]`.
    write_vec : `[batch_size, word_size]`.

    Returns
    -------
    mem : `[batch_size, mem_size, word_size]`.
    """
    add_content = 1 - expand_dims(ww, axis=2) @ expand_dims(erase, axis=1)
    forget = expand_dims(ww, axis=2) @ expand_dims(write_vec, axis=1)
    return mem_tm1 * forget + add_content


class NeuralCPU(object):
    """Neural CPU class"""

    def __init__(self, input_size, output_size, mem_size, word_size, n_reads):
        self.params = collections.OrderedDict.fromkeys([
            'W_f', 'W_i', 'W_c', 'W_o',
            'b_f', 'b_i', 'b_c', 'b_o',
            'W_y', 'W_r', 'W_xi'
        ])
        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.word_size = word_size
        self.n_readheads = n_reads

        output_sz = self.output_size
        h_sz = self.input_size
        x_sz = self.input_size + h_sz + self.n_readheads * self.word_size
        wr_sz = self.n_readheads * self.word_size
        wxi_sz = (self.word_size * self.n_readheads + self.word_size * 3 +
                  self.n_readheads * 5 + 3)

        self.params['W_f'] = get_xavier_weights(x_sz, h_sz, name='W_f')
        self.params['W_i'] = get_xavier_weights(x_sz, h_sz, name='W_i')
        self.params['W_c'] = get_xavier_weights(x_sz, h_sz, name='W_c')
        self.params['W_o'] = get_xavier_weights(x_sz, h_sz, name='W_o')

        self.params['b_f'] = get_zero_weights(1, h_sz, name='b_f')
        self.params['b_i'] = get_zero_weights(1, h_sz, name='b_i')
        self.params['b_c'] = get_zero_weights(1, h_sz, name='b_c')
        self.params['b_o'] = get_zero_weights(1, h_sz, name='b_o')

        self.params['W_y'] = get_rand_weights(h_sz, output_sz, name='W_y')
        self.params['W_r'] = get_rand_weights(wr_sz, output_sz, name='W_r')
        self.params['W_xi'] = get_rand_weights(h_sz, wxi_sz, name='W_xi')

    def step(self, x, h_tm1, c_tm1, r_tm1_ls,
             mem_tm1, tpmem_tm1, ww_tm1, wr_tm1_ls,
             u_tm1, p_tm1):
        """
        Parameters
        ----------
        x : `[batch_size, input_size]`
        h_tm1 : `[batch_size, input_size]`
        c_tm1 : `[batch_size, input_size]`
        r_tm1_ls : read vectors list, each element in the list has size of:
            `[batch_size, word_size]`
        mem_tm1 : `[batch_size, word_size, mem_size]`.
        tpmem_tm1 : `[batch_size, mem_size, mem_size]`.
        ww_tm1 : `[batch_size, mem_size]`.
        wr_tm1_ls : each element in the list has size of:
            `[batch_size, mem_size]`.
        u_tm1 : `[batch_size, mem_size]`.
        p_tm1 : `[batch_size, mem_size]`.

        Returns
        -------
        """
        x_ = concat([x, h_tm1] + r_tm1_ls, axis=1)

        f = sigmoid(x_ @ self.params['W_f'] + self.params['b_f'])
        i = sigmoid(x_ @ self.params['W_i'] + self.params['b_i'])
        o = sigmoid(x_ @ self.params['W_o'] + self.params['b_o'])
        c_ = tanh(x_ @ self.params['W_c'] + self.params['b_c'])
        c = f * c_tm1 + i * c_
        h = o * tanh(c)

        xi = h @ self.params['W_xi']
        intfc = intfcparse(xi, self.word_size, self.n_readheads)

        allocvec, u = allocmem(u_tm1, ww_tm1, wr_tm1_ls, intfc['fg'])
        content_w = contentaddress(intfc['key_w'], mem_tm1, intfc['beta_w'])
        ww = writeweight(allocvec, content_w, intfc['g_w'], intfc['g_a'])
        mem = memwrite(mem_tm1, ww, intfc['e'], intfc['v'])

        tpmem, p, fseq_ls, bseq_ls = tpmemrecall(tpmem_tm1, p_tm1, ww,
                                                 wr_tm1_ls)
        content_r_ls = [contentaddress(k_r, mem, b_r)
                        for k_r, b_r in zip(intfc['key_r'], intfc['beta_r'])]
        wr_ls = readweight(fseq_ls, bseq_ls, content_r_ls, intfc['pi'])
        r_ls = memread(mem, wr_ls)

        y = h @ self.params['W_y'] + concat(r_ls, axis=1) @ self.params['W_r']

        return y, h, c, r_ls, mem, tpmem, ww, wr_ls, u, p
