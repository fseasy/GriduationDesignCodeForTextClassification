#/usr/bin/env python
#coding=utf-8

import sys
import os
import argparse
import logging
import re
import numpy as np
import xgboost as xgt
import traceback
try :
    import cPickle as pickle
except :
    import pickle


from collections import Counter
from file_handler import *

### should be specify
POSITIVE_LABEL=1
NEGATIVE_LABEL=0 

logging.basicConfig(level=logging.DEBUG)
def build_SVM_format_X_from_docs_words(doc_words_list , docs_dict) :
    '''
    input > doc_words_list : [ [ (word , TF),... ],... ]
    return > [ {idx:val , ...} , ...  ]
    ###ATTENTION ! SVM format , idx counted from 1 , not 0
    '''
    X = []
    query_dict = dict(zip(docs_dict , range(1,len(docs_dict) + 1)))
    for one_docs in doc_words_list :
        #[ (word , TF)]
        one_docs_repr = {}
        for word , tf in one_docs :
            if word in query_dict :
                idx_svm = query_dict[word]
                one_docs_repr[idx_svm] = 1
        X.append(one_docs_repr)
    return X


def main(postrain_f , negtrain_f , postest_f , negtest_f , gram_n , otrain_f , otest_f ) :
    logging.info("statistic docs info")
    pos_train_docs , pos_train_labels , pos_train_words_counter = stat_oneclass_docs(postrain_f , POSITIVE_LABEL , gram_n)
    neg_train_docs , neg_train_labels , neg_train_words_counter = stat_oneclass_docs(negtrain_f , NEGATIVE_LABEL , gram_n)
    pos_test_docs , pos_test_labels , pos_test_words_counter = stat_oneclass_docs(postest_f , POSITIVE_LABEL , gram_n)
    neg_test_docs , neg_test_labels , neg_test_words_counter = stat_oneclass_docs(negtest_f , NEGATIVE_LABEL , gram_n)

    logging.info("build dict")
    docs_dict = build_docs_dict([pos_train_words_counter , neg_train_words_counter])

    #READY X,Y for training
    logging.info("ready trainning data")
    Y = []
    Y.extend(pos_train_labels)
    Y.extend(neg_train_labels)
    
    train_docs = []
    train_docs.extend(pos_train_docs)
    train_docs.extend(neg_train_docs)
    X = build_SVM_format_X_from_docs_words(train_docs , docs_dict)
    logging.info("write trainning data to '%s'" %(otrain_f.name))
    save_in_libsvm_sparse_format(Y,X,otrain_f) 
    
    logging.info("ready testing data")
    Y = []
    Y.extend(pos_test_labels)
    Y.extend(neg_test_labels)

    test_docs = []
    test_docs.extend(pos_test_docs)
    test_docs.extend(neg_test_docs)
    X = build_SVM_format_X_from_docs_words(test_docs  , docs_dict)
    logging.info("write testing data to '%s'" %(otest_f.name))
    save_in_libsvm_sparse_format(Y,X,otest_f)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="[using bool feature] trans training and  testing corpus to the libsvm format data")
    argp.add_argument("-ptrain" , "--postrain" , help="path to positive training corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-ptest" , "--postest" , help="path to positive testing corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-ntrain" , "--negtrain" , help="path to negative training corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-ntest" , "--negtest" , help="path to negative testing corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-g" , "--gram" , help="ngram num" , choices=[1,2,3] , required=True , type=int)
    argp.add_argument("-o" , "--output_prefix" , help="ouput file path(will be added .train and .test)" , required=True , type=str)

    args = argp.parse_args()
    
    output_dir , output_name = os.path.split(args.output_prefix)
    if len(output_dir) > 0 and not os.path.exists(output_dir) :
        try :
            os.makedirs(output_dir)
        except :
            traceback.print_exc()
            exit(1)
    otrain_f = open(args.output_prefix + '.train' , 'w')
    otest_f = open(args.output_prefix + '.test' , 'w')
    
    main(args.postrain ,args.negtrain , args.postest , args.negtest , args.gram , otrain_f , otest_f)

    args.postrain.close()
    args.negtrain.close()
    args.postest.close()
    args.negtest.close()

    otrain_f.close()
    otest_f.close()
