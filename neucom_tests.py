#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Wan Xiaolin <wanxl13@lzu.edu.cn>

"""Test for neural computer.

"""

import collections
import re

import numpy as np
import tensorflow as tf

import neucom
from neucom import DTYPE

BATCH_SIZE = 2
N_READS = 3
MEM_SIZE = 10
WORD_SIZE = 5
INPUT_SIZE = 4
OUTPUT_SIZE = 6


def get_pairs(name, shape, n_reads):
    if re.search('_ls\Z', name):
        k = [tf.placeholder(DTYPE, shape, name=name[:-2] + '%d' % i)
             for i in range(n_reads)]
        v = [np.random.random_sample(shape).astype(DTYPE)
             for _ in range(n_reads)]
        return collections.OrderedDict(zip(k, v))
    else:
        k = tf.placeholder(DTYPE, shape, name=name)
        v = np.random.random_sample(shape).astype(DTYPE)
        return collections.OrderedDict([(k, v)])


def getvars(names: list, shapes: list, n_reads):
    output = [{}, ]
    for n, s in zip(names, shapes):
        d = get_pairs(n, s, n_reads=n_reads)
        output[0].update(d)
        dkeys = list(d.keys())
        if len(dkeys) == 1:
            output.append(dkeys[0])
        else:
            output.append(dkeys)
    return output


class NCPUModuleBuildTest(tf.test.TestCase):

    def test_intfcparse(self):
        names = ['xi']
        shapes = [
            [BATCH_SIZE, WORD_SIZE*N_READS + WORD_SIZE*3 + N_READS*5 + 3],
        ]
        feed, xi = getvars(names, shapes, N_READS)
        y = neucom.intfcparse(xi, WORD_SIZE, N_READS)

        with self.test_session() as sess:
            intfc = sess.run(y, feed)

        self.assertEqual(intfc['key_r'][0].shape, (BATCH_SIZE, WORD_SIZE))
        self.assertEqual(intfc['beta_r'][0].shape, (BATCH_SIZE, 1))
        self.assertEqual(intfc['key_w'].shape, (BATCH_SIZE, WORD_SIZE))
        self.assertEqual(intfc['beta_w'].shape, (BATCH_SIZE, 1))
        self.assertEqual(intfc['e'].shape, (BATCH_SIZE, WORD_SIZE))
        self.assertEqual(intfc['v'].shape, (BATCH_SIZE, WORD_SIZE))
        self.assertEqual(intfc['fg'][0].shape, (BATCH_SIZE, 1))
        self.assertEqual(intfc['g_a'].shape, (BATCH_SIZE, 1))
        self.assertEqual(intfc['g_w'].shape, (BATCH_SIZE, 1))
        self.assertEqual(intfc['pi'][0].shape, (BATCH_SIZE, 3))

        self.assertEqual(len(intfc['key_r']), N_READS)

    def test_contentaddress(self):
        names = ['key', 'mem', 'beta']
        shapes = [
            [BATCH_SIZE, WORD_SIZE],
            [BATCH_SIZE, MEM_SIZE, WORD_SIZE],
            [BATCH_SIZE, 1]
        ]
        feed, key, mem, beta = getvars(names, shapes, N_READS)
        y = neucom.contentaddress(key, mem, beta)

        with self.test_session() as sess:
            rval = sess.run(y, feed)

        self.assertEqual(rval.shape, (BATCH_SIZE, MEM_SIZE))

    def test_allocmem(self):
        names = ['u_tm1', 'ww_tm1', 'wr_tm1_ls', 'fg_ls']
        shapes = [
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, 1]
        ]
        feed, u_tm1, ww_tm1, wr_tm1_ls, fg_ls = getvars(names, shapes, N_READS)
        y = neucom.allocmem(u_tm1, ww_tm1, wr_tm1_ls, fg_ls)

        with self.test_session() as sess:
            alloc_vec, u = sess.run(y, feed)

        self.assertEqual(alloc_vec.shape, (BATCH_SIZE, MEM_SIZE))
        self.assertEqual(u.shape, (BATCH_SIZE, MEM_SIZE))

    def test_tpmemrecall(self):
        names = ['tpmem_tm1', 'p_tm1', 'ww', 'wr_tm1_ls']
        shapes = [
            [BATCH_SIZE, MEM_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE]
        ]
        feed, tpmem_tm1, p_tm1, ww, wr_tm1_ls = getvars(names, shapes, N_READS)
        y = neucom.tpmemrecall(tpmem_tm1, p_tm1, ww, wr_tm1_ls)

        with self.test_session() as sess:
            tpmem, p, fseq_ls, bseq_ls = sess.run(y, feed)

        self.assertEqual(tpmem.shape, (BATCH_SIZE, MEM_SIZE, MEM_SIZE))
        self.assertEqual(p.shape, (BATCH_SIZE, MEM_SIZE))
        self.assertEqual(fseq_ls[0].shape, (BATCH_SIZE, MEM_SIZE))
        self.assertEqual(bseq_ls[0].shape, (BATCH_SIZE, MEM_SIZE))

    def test_readweight(self):
        names = ['fseq_ls', 'bseq_ls', 'content_r_ls', 'pi_ls']
        shapes = [
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, 3]
        ]
        feed, fseq_ls, bseq_ls, content_r_ls, pi_ls = getvars(names,
                                                              shapes, N_READS)
        y = neucom.readweight(fseq_ls, bseq_ls, content_r_ls, pi_ls)

        with self.test_session() as sess:
            wr_ls = sess.run(y, feed)

        self.assertEqual(wr_ls[0].shape, (BATCH_SIZE, MEM_SIZE))

    def test_writeweight(self):
        names = ['alloc', 'content_w', 'g_w', 'g_a']
        shapes = [
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, 1],
            [BATCH_SIZE, 1]
        ]
        feed, alloc, content_w, g_w, g_a = getvars(names, shapes, N_READS)
        y = neucom.writeweight(alloc, content_w, g_w, g_a)

        with self.test_session() as sess:
            ww = sess.run(y, feed)

        self.assertEqual(ww.shape, (BATCH_SIZE, MEM_SIZE))

    def test_memread(self):
        names = ['mem', 'wr_ls']
        shapes = [
            [BATCH_SIZE, MEM_SIZE, WORD_SIZE],
            [BATCH_SIZE, MEM_SIZE]
        ]
        feed, mem, wr_ls = getvars(names, shapes, N_READS)
        y = neucom.memread(mem, wr_ls)

        with self.test_session() as sess:
            r_ls = sess.run(y, feed)

        self.assertEqual(r_ls[0].shape, (BATCH_SIZE, WORD_SIZE))

    def test_memwrite(self):
        names = ['mem_tm1', 'ww', 'erase', 'write_vec']
        shapes = [
            [BATCH_SIZE, MEM_SIZE, WORD_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, WORD_SIZE],
            [BATCH_SIZE, WORD_SIZE]
        ]
        feed, mem_tm1, ww, erase, write_vec = getvars(names, shapes, N_READS)
        y = neucom.memwrite(mem_tm1, ww, erase, write_vec)

        with self.test_session() as sess:
            mem = sess.run(y, feed)

        self.assertEqual(mem.shape, (BATCH_SIZE, MEM_SIZE, WORD_SIZE))


class NeuralCPUBuildTest(tf.test.TestCase):

    def test_step(self):
        model = neucom.NeuralCPU(INPUT_SIZE, OUTPUT_SIZE, MEM_SIZE, WORD_SIZE,
                                 N_READS)
        names = [
            'x', 'h_tm1', 'c_tm1', 'r_tm1_ls',
            'mem_tm1', 'tpmem_tm1', 'ww_tm1', 'wr_tm1_ls',
            'u_tm1', 'p_tm1'
        ]
        shapes = [
            [BATCH_SIZE, INPUT_SIZE],
            [BATCH_SIZE, INPUT_SIZE],
            [BATCH_SIZE, INPUT_SIZE],
            [BATCH_SIZE, WORD_SIZE],

            [BATCH_SIZE, MEM_SIZE, WORD_SIZE],
            [BATCH_SIZE, MEM_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE],

            [BATCH_SIZE, MEM_SIZE],
            [BATCH_SIZE, MEM_SIZE]
        ]

        feed, x, h_tm1, c_tm1, r_tm1_ls, \
            mem_tm1, tpmem_tm1, ww_tm1, wr_tm1_ls, \
            u_tm1, p_tm1 \
            = getvars(names, shapes, N_READS)

        rval = model.step(x, h_tm1, c_tm1, r_tm1_ls,
                          mem_tm1, tpmem_tm1, ww_tm1, wr_tm1_ls,
                          u_tm1, p_tm1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y, h, c, r_ls, mem, tpmem, ww, wr_ls, u, p = sess.run(rval, feed)

        self.assertEqual(y.shape, (BATCH_SIZE, OUTPUT_SIZE))
        self.assertEqual(h.shape, (BATCH_SIZE, INPUT_SIZE))
        self.assertEqual(c.shape, (BATCH_SIZE, INPUT_SIZE))
        self.assertEqual(r_ls[0].shape, (BATCH_SIZE, WORD_SIZE))

        self.assertEqual(mem.shape, (BATCH_SIZE, MEM_SIZE, WORD_SIZE))
        self.assertEqual(tpmem.shape, (BATCH_SIZE, MEM_SIZE, MEM_SIZE))
        self.assertEqual(ww.shape, (BATCH_SIZE, MEM_SIZE))
        self.assertEqual(wr_ls[0].shape, (BATCH_SIZE, MEM_SIZE))

        self.assertEqual(u.shape, (BATCH_SIZE, MEM_SIZE))
        self.assertEqual(p.shape, (BATCH_SIZE, MEM_SIZE))
