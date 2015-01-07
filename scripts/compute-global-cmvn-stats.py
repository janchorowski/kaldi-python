#!/usr/bin/env python

'''
Created on Aug 6, 2014

@author: chorows
'''

import os
import sys
import logging
import argparse

import numpy as np

import kaldi_io

if __name__ == '__main__':
    print >>sys.stderr, os.path.basename(sys.argv[0]), " ".join(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Accumulate global stats for feature normalization: mean and std')
    parser.add_argument('in_rxfilename')
    parser.add_argument('out_wxfilename')
    args = parser.parse_args()
    
    sum = None
    sum_sq = None
    n = 0
    
    with kaldi_io.SequentialBaseFloatMatrixReader(args.in_rxfilename) as reader:
        for name,feats in reader:
            nframes, nfeats = feats.shape
            n += nframes
            if sum is None:
                sum = np.zeros((nfeats,))
                sum_sq = np.zeros((nfeats,))
                
            sum += feats.sum(0)
            sum_sq += (feats*feats).sum(0) 
    
    mean = np.asarray(sum/n, dtype=kaldi_io.KALDI_BASE_FLOAT())
    std = np.asarray(np.sqrt(sum_sq/n - mean**2), 
                     dtype=kaldi_io.KALDI_BASE_FLOAT())
    
    with kaldi_io.BaseFloatVectorWriter(args.out_wxfilename) as w:
        w['mean'] = mean
        w['std'] = std
