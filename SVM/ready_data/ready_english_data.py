#!/usr/bin/env python
#coding=utf-8

import logging
import os
import sys
import argparse
import re
import string

#logging.basicConfig(level=logging.DEBUG)

filter_str = string.punctuation + string.digits 
trans_table = string.maketrans(filter_str , ' '*len(filter_str))

def tokenize(l) :
    l = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])","",l) # remove the `'` of in the words , such as can't and so on , avoid repalce by space at follow
    l = string.translate(l,trans_table)
    tokens = l.strip().lower().split() # lower , the capitalize info may be no use for classfication
    return filter(lambda x : re.match(r'^[a-z]+$' , x) != None , tokens)

def process_input_as_dir(inpath , outpath) :
    flist = os.listdir(inpath)
    of = open(outpath , 'w')
    for f in flist :
        path = os.path.join(inpath , f)
        with open(path) as f_i :
            tokens = []
            for l in f_i :
                tokens.extend(tokenize(l))
            logging.debug('file:%s , tokens: %s' %(path , ' '.join(tokens)))
            of.write(' '.join(tokens) + '\n') # one file corresponding to one line 
    of.close()

def process_input_as_file(inpath , outpath) :
    of = open(outpath , 'w')
    with open(inpath) as f_i :
        for l in f_i :
            tokens = tokenize(l)
            of.write(' '.join(tokens) + '\n') # one line is a file , just remove the char that does not needed
    of.close()

def main(inpath , outpath) :
    '''
    input > inpath : dir path to resources 
            outpath : output path , check for exits , avoid rewrite
    return < none
    read all files , and for every file , write to the result as one line , splited with blank . and , only leave te [A-Za-z] pattern
    '''
    if not os.path.exists(inpath) :
        logging.error('"%s" does not exists' %(inpath))
        return 0 
    if os.path.exists(outpath) :
        r = raw_input("outpath '%s' has already exists! \n rewrite[y/n] ?" %(outpath))
        if r != "y" :
            return 0
    if os.path.isfile(inpath) :
        process_input_as_file(inpath , outpath)
    else :
        process_input_as_dir(inpath , outpath)

if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(description="ready data of english")
    argparser.add_argument('--inpath' , help="path to raw sources dir" , type=str , required=True)
    argparser.add_argument('--outpath' , help="path to output result " , type=str , required=True)

    args = vars(argparser.parse_args())
    main(**args)
