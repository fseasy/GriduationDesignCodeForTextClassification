#/usr/bin/env python
#coding=utf-8

import logging
import sys
import numpy as np

logging.basicConfig(level=logging.INFO)

def load_wordvector(wordvector_f) :
    logging.info("loading word vector.")
    vector_infos = wordvector_f.readline().rstrip().split()
    assert(len(vector_infos) == 2)
    logging.info("vector num : %s , dimention : %s \n" %(vector_infos[0] , vector_infos[1]))
    wordvectors = {}
    load_idx = 0
    for line in wordvector_f :
        parts = line.rstrip().split()
        word = parts[0]
        vector = parts[1:]
        vector = [float(x) for x in vector]
        vector = np.mat(vector)
        wordvectors[word] = vector
        load_idx += 1
        if load_idx %1000 == 0 :
            print >> sys.stderr , "\rloaded %-10d word vectors" %(load_idx) ,
    print >> sys.stderr , "\n"
    logging.info("loading word vector done.")
    return wordvectors

