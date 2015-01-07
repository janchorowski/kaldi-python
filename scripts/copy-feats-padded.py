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
    
    parser = argparse.ArgumentParser(description='Copy features, additionally padding them (elongating them) by the specified number of frames')
    parser.add_argument('--padding', help='With how many frames to pad on each side? [default=0]', default=None)
    parser.add_argument('--padding-left', help='With how many frames to pad on the left (defaults to padding if not set)', default=None)
    parser.add_argument('--padding-right', help='With how many frames to pad on the right (defaults to padding if not set)', default=None)
    parser.add_argument('--padding-mode', default='zero', help='What values to use for padding- zero|copy (edge frames)')
    parser.add_argument('--orig-size-wxfilename', help='Where to write the original matrix sizes', default=None)
    parser.add_argument('in_rxfilename')
    parser.add_argument('out_wxfilename')
    args = parser.parse_args()
    
    if args.padding is not None and args.padding_left is not None and args.padding_right is not None:
        logging.error("Can't set padding, padding-left and padding-right at the same time!")
        sys.exit(1)
    
    padding = 0
    if args.padding is not None: padding = int(args.padding)
    
    padding_left = padding
    if args.padding_left is not None: padding_left = int(args.padding_left)
    
    padding_right = padding
    if args.padding_right is not None: padding_right = int(args.padding_right)
    
    if padding_left<0 or padding_right<0:
        logging.error("Padding can't be negative!")
        sys.exit(1)
    
    count = 0
    logging.info("Padding with %d in the left and %d on the right", padding_left, padding_right)
    
    #should use with, but if something happens the files will get closed anyways
    reader = kaldi_io.SequentialBaseFloatMatrixReader(args.in_rxfilename)
    writer = kaldi_io.BaseFloatMatrixWriter(args.out_wxfilename)
    
    size_writer=None
    if args.orig_size_wxfilename is not None:
        size_writer = kaldi_io.PythonWriter(args.orig_size_wxfilename)
    
    for name, value in reader:
        count += 1
        if padding_left+padding_right==0:
            padded = value
        else:
            num_frames, frame_dim = value.shape
            padded = np.empty(shape=(num_frames+padding_left+padding_right, frame_dim), dtype=value.dtype)
            
            padded[padding_left:padding_left+num_frames,:] = value
            
            if args.padding_mode == 'zero':
                padded[:padding_left,:] = 0.0
                padded[padding_left+num_frames:,:] = 0.0
            elif args.padding_mode == 'copy':
                padded[:padding_left,:] = value[0,:]
                padded[padding_left+num_frames:,:] = value[-1,:]
            else:
                logging.error("Unknown padding mode: %s", args.padding_mode)
                sys.exit(1)
        writer.write(name, padded)
        if size_writer:
            size_writer.write(name, value.shape)

    logging.info("Copied %d features", count)
