#!/usr/bin/env python
'''
Created on Jul 31, 2014

@author: chorows
'''

import sys
import argparse
import tempfile
from subprocess import check_call
import os
from os import path

import numpy as np

import kaldi_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract an utterance and convert the alignment to an Audacity label file')
    parser.add_argument('wav', help='wav script file')
    parser.add_argument('mdl', help='model file (to get transitions)')
    parser.add_argument('ali', help='alignemnt')
    parser.add_argument('phn', help='phones.txt')
    parser.add_argument('utt', help='utterance')
    args = parser.parse_args()
    
    #temp_dir = tempfile.mkdtemp()
    temp_dir = './tmp'
    try:
        os.mkdir(temp_dir)
    except:
        pass
    
    utt=args.utt
    
    wav_file = path.join(temp_dir, '%s.wav' %(utt,))
    print >>sys.stderr, "Extracting wav utterance %s" % (utt,)
    check_call("wav-copy '%s' 'scp,p:echo %s %s|'" %
         (args.wav, utt, wav_file), shell=True) 
    dur_reader = kaldi_io.RandomAccessPythonReader(
       "ark:wav-to-duration 'scp:echo %s %s |' ark,t:-|" % 
       (utt, wav_file))
    dur = dur_reader[utt]
    ali_reader = kaldi_io.RandomAccessInt32PairVectorReader(
      "ark:ali-to-phones --write-lengths '%s' '%s' 'ark:-' |" %
      (args.mdl, args.ali))
    ali = np.array(ali_reader[utt], dtype=float)
    num_frames = ali[:,1].sum()
    
    ali[:,1] = (np.cumsum(ali[:,1]))/num_frames*dur
    
    phones_dict = {n:p for p,n in kaldi_io.SequentialPythonReader('ark:%s' %(args.phn,))}

    label_file = path.join(temp_dir, '%s.txt'%(utt,))
    last_time = 0.0
    with open(label_file, 'w') as lf:
        for row in ali:
            (phone, time) = row
            print >>lf, '%f %f %s' % (last_time, time, phones_dict[phone])
            last_time=time
    
    
    check_call('audacity %s' % (wav_file,), shell=True)
    shutil.rmdir(temp_dir)
