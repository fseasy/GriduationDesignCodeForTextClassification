#/usr/bin/env python
#coding=utf-8


import sys
import os
import argparse
import logging
import re
import numpy as np
import traceback
try :
    import cPickle as pickle
except :
    import pickle

from collections import Counter
from fileprocessing import *
from evaluate import *

logging.basicConfig(level=logging.DEBUG)

def stat_oneclass_docs(docs_f , docs_label , gram_num) :
    '''
    return > this class' docs words and TF , [ [(wrod , TF) ,] ] 
             class labels , [ label , ..]
             class' all words and 'DF' , Counter
    '''
    class_counter = Counter()
    class_docs = []
    class_labels = []
    for doc in docs_f :
        all_words_list = tokenize(doc , gram_num)
        doc_counter = Counter(all_words_list)
        class_docs.append(doc_counter.items())
        class_counter.update(doc_counter.keys())
        class_labels.append(docs_label)
    return class_docs , class_labels , class_counter

def build_docs_dict(class_counter_list) :
    dict_counter = Counter()
    for item in class_counter_list :
        dict_counter.update(item.keys())
    return dict_counter.keys()

def calc_allwords_df(class_counter_list) :
    words_df = Counter()
    for item in class_counter_list :
        words_df.update(item)
    return words_df

def build_SVM_format_X_from_docs_words(doc_words_list , words_df , docs_dict) :
    '''
    input > doc_words_list : [ [ (word , TF),... ],... ]
    return > [ {idx:val , ...} , ...  ]
    ###ATTENTION ! SVM format , idx counted from 1 , not 0
    '''
    X = []
    docs_num = float(len(doc_words_list))
    query_dict = dict(zip(docs_dict , range(1,len(docs_dict) + 1)))
    for one_docs in doc_words_list :
        #[ (word , TF)]
        one_docs_repr = {}
        square_sum = 0
        for word , tf in one_docs :
            if word in query_dict :
                idx_svm = query_dict[word]
                assert word in words_df
                df = words_df[word]
                idf = np.log( docs_num / df )
                val = tf*idf
                square_sum += pow(val,2)
                if val != 0 :
                    one_docs_repr[idx_svm] = val
        #normalize
        for idx in one_docs_repr :
            try :
                one_docs_repr[idx] /= square_sum # may be zero , leave it
            except ZeroDivisionError , e :
                traceback.print_exc()
        X.append(one_docs_repr)
    return X

def liblinear_train(Y,X,param) :
    c = param['c']
    bias = param['bias']
    w_positive = param['w_positive']
    w_negative = param['w_negative']

    logging.info("")
    logging.info("training using C = %.2f , w_positive = %.2f , w_negative = %.2f , bias = %.2f\n" %(c , w_positive , w_negative ,bias))
    params_str = "-c %.2f -w%s %.2f -w%s %.2f -B %.2f -q" %( c , POSITIVE_LABEL , w_positive , NEGATIVE_LABEL , w_negative , bias  )
    prob = liblinearutil.problem(Y,X)
    param = liblinearutil.parameter(params_str)
    model = liblinearutil.train(prob , param)
    return model



def main(postrain_f , negtrain_f , postest_f , negtest_f , gram_n , param) :
    logging.info("statistic docs info")
    pos_train_docs , pos_train_labels , pos_train_words_counter = stat_oneclass_docs(postrain_f , POSITIVE_LABEL , gram_n)
    neg_train_docs , neg_train_labels , neg_train_words_counter = stat_oneclass_docs(negtrain_f , NEGATIVE_LABEL , gram_n)
    pos_test_docs , pos_test_labels , pos_test_words_counter = stat_oneclass_docs(postest_f , POSITIVE_LABEL , gram_n)
    neg_test_docs , neg_test_labels , neg_test_words_counter = stat_oneclass_docs(negtest_f , NEGATIVE_LABEL , gram_n)

    logging.info("build dict")
    docs_dict = build_docs_dict([pos_train_words_counter , neg_train_words_counter])
    
    words_df = calc_allwords_df([pos_train_words_counter , neg_train_words_counter])

    #READY X,Y for training
    logging.info("ready trainning data")
    Y = []
    Y.extend(pos_train_labels)
    Y.extend(neg_train_labels)
    
    train_docs = []
    train_docs.extend(pos_train_docs)
    train_docs.extend(neg_train_docs)
    X = build_SVM_format_X_from_docs_words(train_docs , words_df , docs_dict)
    logging.info("training using liblinear")
    model = liblinear_train(Y,X,param)
    logging.info("ready testing data")
    Y = []
    Y.extend(pos_test_labels)
    Y.extend(neg_test_labels)

    test_docs = []
    test_docs.extend(pos_test_docs)
    test_docs.extend(neg_test_docs)
    X = build_SVM_format_X_from_docs_words(test_docs , words_df , docs_dict)
    logging.info("predict using liblinear")
    p_labels , p_acc , p_val = liblinearutil.predict(Y,X,model,"-q")
    positive_prf , negative_prf = calc_prf(Y,p_labels)
    positive_prf = map(lambda x:100*x , positive_prf)
    negative_prf = map(lambda x:100*x , negative_prf)
    print "accuracy = %.2f %%" %(p_acc[0])
    print "positive class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( positive_prf[0] , positive_prf[1] , positive_prf[2])
    print "negative class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( negative_prf[0] , negative_prf[1] , negative_prf[2])
    

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="svm with tf-idf feature  test ")
    argp.add_argument("-ptrain" , "--postrain" , help="path to positive training corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-ptest" , "--postest" , help="path to positive testing corpus" , required=True , type=argparse.FileType('r'))

    argp.add_argument("-ntrain" , "--negtrain" , help="path to negative training corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-ntest" , "--negtest" , help="path to negative testing corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-g" , "--gram" , help="ngram num" , choices=[1,2,3] , required=True , type=int)
    ##liblinear parameter
    argp.add_argument("-c" , "--c" , help="liblinear parameter C " , required=True , type=float)
    argp.add_argument("-b" , "--bias",help="liblinear parameter B " , required=True , type=float)
    argp.add_argument("-w_p" , "--w_positive" , help="liblinear parameter wi for positive" , required=True , type=float)
    argp.add_argument("-w_n" , "--w_negative" , help="liblinear parameter wi for negative" , required=True , type=float)
    
    argp.add_argument("--liblinear" , help="path liblinear python interface lib" , default="/users1/wxu/bin/liblinear-1.96/python")

    args = argp.parse_args()
    try :
        sys.path.append(args.liblinear)
        import liblinearutil
    except Exception , e :
        logging.error(e)
        exit(1)
    #param preprocess
    liblinear_param = {'bias':args.bias , 'c':args.c , 'w_positive':args.w_positive , 'w_negative':args.w_negative }
    
    main(args.postrain , args.negtrain , args.postest , args.negtest , args.gram , liblinear_param)

    args.postrain.close()
    args.postest.close()
    args.negtrain.close()
    args.negtest.close()
