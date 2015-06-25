#/usr/bin/env python
#coding=utf-8

import sys
import os
import argparse
import logging
import re
import numpy as np
import traceback
from collections import Counter

cur_dir_path= os.path.split(os.path.abspath(__file__))[0]
add_path= os.path.join(cur_dir_path , "../BOOSTEDTREE")
sys.path.append(add_path)

from evaluate import *
from file_handler import load_for_libsvm_sparse_format as load_data , save_in_libsvm_sparse_format as save_data
'''
X,Y load_data(ifi)
    save_data(Y,X,ofi)
'''

logging.basicConfig(level=logging.DEBUG)

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

def compute_NBSVM_param(m , beta=0.25) :
    '''
    intput > m : model of libLinear
             beta : smoothing parameter
    return > w : weight vector
             b : bias
    w' = (1 - beta)*w_mean + beta*w
    where w_mean = norm1(w) / size(V) , V is the feature space
    '''
    dimension = m.get_nr_feature()
    labels = m.get_labels()
    #we need get the label idx for positive label
    label_idx = 0 
    for i in range(0,len(labels)) :
        if labels[i] == POSITIVE_LABEL :
            label_idx = i
            break

    w , b = m.get_decfun(label_idx)
    w = np.array(w)
    w_norm1 = abs(w).sum()
    w_mean = w_norm1 / dimension 
    w_new = (1 - beta) * w_mean * np.ones(dimension) + beta * w 
    return list(w_new) , b

def NBSVM_predict(w,b,test_data_list) :
    w_dimension = len(w)
    w = np.array(w)
    w.shape = (w.shape[0],1) #transpose
    py = []
    for test_data in test_data_list :
        v = np.zeros(w_dimension)
        for idx , val in test_data.items() :
            idx -= 1
            if idx < w_dimension : # because in some case , no values at the end of some colums in the trainning data . so w abandon this
                v[idx] = val
        py.append(1 if (np.dot(v,w) + b) > 0 else 0 )
    return py

def main(train_f , test_f , param) :

    #READY X,Y for training
    logging.info("loading trainning data")
    X , Y = load_data(train_f)
    
    logging.info("training using liblinear")
    model = liblinear_train(Y,X,param)
    
    logging.info('build nbsvm model')
    w , b = compute_NBSVM_param(model , param['beta'])

    logging.info("loading testing data")
    X , Y = load_data(test_f)
    
    logging.info("predict using nbsvm")
    p_labels  = NBSVM_predict(w,b,X)
    positive_prf , negative_prf = calc_prf(Y,p_labels)
    positive_prf = map(lambda x:100*x , positive_prf)
    negative_prf = map(lambda x:100*x , negative_prf)
    acc = calc_acc(Y,p_labels)
    print "accuracy = %.2f %%" %(acc)
    print "positive class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( positive_prf[0] , positive_prf[1] , positive_prf[2])
    print "negative class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( negative_prf[0] , negative_prf[1] , negative_prf[2])
    

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="svm with tf-idf feature  test ")
    argp.add_argument("-train" , "--train" , help="path to training data of sparse libsvm format" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-test" , "--test" , help="path to testing data of sparse libsvm format" , required=True , type=argparse.FileType('r'))
    ##liblinear parameter
    argp.add_argument("-c" , "--c" , help="liblinear parameter C " , required=True , type=float)
    argp.add_argument("-b" , "--bias",help="liblinear parameter B " , required=True , type=float)
    argp.add_argument("-w_p" , "--w_positive" , help="liblinear parameter wi for positive" , required=True , type=float)
    argp.add_argument("-w_n" , "--w_negative" , help="liblinear parameter wi for negative" , required=True , type=float)
    
    argp.add_argument("-beta" , "--beta",help="NBSVM parameter beta " , required=True , type=float)
    
    argp.add_argument("--liblinear" , help="path liblinear python interface lib" , default="/users1/wxu/bin/liblinear-1.96/python")

    args = argp.parse_args()
    try :
        sys.path.append(args.liblinear)
        import liblinearutil
    except Exception , e :
        logging.error(e)
        exit(1)
    #param preprocess
    liblinear_param = {'bias':args.bias , 'c':args.c , 'w_positive':args.w_positive , 'w_negative':args.w_negative , 'beta':args.beta }
    
    main(args.train , args.test , liblinear_param)

    args.train.close()
    args.test.close()
