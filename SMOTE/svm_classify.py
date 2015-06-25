#/usr/bin/env python
#coding=utf-8


import sys
import os
import argparse
import logging
import re
import numpy as np
import traceback

cur_dir_path= os.path.split(os.path.abspath(__file__))[0]
add_path= os.path.join(cur_dir_path , "../BOOSTEDTREE")
sys.path.append(add_path)

from collections import Counter
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
    logging.info("training using C = %f , w_positive = %.2f , w_negative = %.2f , bias = %.2f\n" %(c , w_positive , w_negative ,bias))
    params_str = "-c %f -w%s %.2f -w%s %.2f -B %.2f -q" %( c , POSITIVE_LABEL , w_positive , NEGATIVE_LABEL , w_negative , bias  )
    prob = liblinearutil.problem(Y,X)
    param = liblinearutil.parameter(params_str)
    model = liblinearutil.train(prob , param)
    return model



def main(train_f , test_f , param) :

    #READY X,Y for training
    logging.info("loading trainning data")
    X , Y = load_data(train_f)
    
    logging.info("training using liblinear")
    model = liblinear_train(Y,X,param)
    
    logging.info("ready testing data")
    X , Y = load_data(test_f)
    
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
    argp.add_argument("-train" , "--train" , help="path to training data of sparse libsvm format" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-test" , "--test" , help="path to testing data of sparse libsvm format" , required=True , type=argparse.FileType('r'))
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
    
    main(args.train , args.test , liblinear_param)

    args.train.close()
    args.test.close()
