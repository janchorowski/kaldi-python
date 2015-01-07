#!/usr/bin/env python

'''
Created on Aug 6, 2014

@author: chorows
'''

import sys
import os
import logging
import argparse

import numpy as np

import kaldi_io

if __name__ == '__main__':
    print >>sys.stderr, os.path.basename(sys.argv[0]), " ".join(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="""Apply cmvn (cepstral mean and variance normalization).
    
    If global stats are supplied, normalize by the global stats. Otherwise normalize per-utterance.
    """, )
    parser.add_argument('--global-stats', help="Global normalization stats")
    parser.add_argument('in_rxfilename')
    parser.add_argument('out_wxfilename')
    args = parser.parse_args()
    
    global_normalization = args.global_stats is not None
    
    
    if global_normalization:
        logging.info("Applying global normalization")
        with kaldi_io.RandomAccessBaseFloatVectorReader(args.global_stats) as stats:
            mean = stats['mean']
            mean.shape = 1,-1
            std = stats['std']
            std.shape = 1,-1
    else:
        logging.info("Applying per-utterance normalization")
    
    reader = kaldi_io.SequentialBaseFloatMatrixReader(args.in_rxfilename)
    writer = kaldi_io.BaseFloatMatrixWriter(args.out_wxfilename)
    
    for name, feats in reader:
        if not global_normalization:
            mean = feats.mean(0, keepdims=True)
            std = feats.std(0, keepdims=True)
        feats -= mean
        feats /= std
        writer.write(name, feats)
