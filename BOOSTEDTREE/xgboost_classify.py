#/usr/bin/env python
#coding=utf8

import xgboost as xgb
import argparse
import logging
import os
import sys

abs_path = os.path.split(os.path.abspath(__file__))[0]
NBSVM_path = os.path.normpath(abs_path + "/../SVM")
sys.path.append(NBSVM_path)

from evaluate import calc_prf , calc_acc

logging.basicConfig(level=logging.DEBUG)


OUT_POSITIVE_LABEL = 1
OUT_NEGATIVE_LABEL = -1

TRAIN_POSITIVE_LABEL = 1
TRAIN_NEGATIVE_LABEL = 0
def main(train_path , test_path ) :
    
    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)
    params = {'bst:max_depth':2 , 'bst:eta':1 , 'silent':1 , 'objective':'binary:logistic' , 'nthread':6}
    params_list = params.items()
    #watch_list = [(dtrain , 'train')] 
    num_round = 2000
    bst = xgb.train(params_list , dtrain , num_round)
    
    ypred_prob = bst.predict(dtest)

    threshold = 0.5 
    ypred = [ p >= 0.5 and OUT_POSITIVE_LABEL or OUT_NEGATIVE_LABEL for p in ypred_prob ]
    y = dtest.get_label()
    #trans label
    y = [ l == TRAIN_POSITIVE_LABEL and OUT_POSITIVE_LABEL or OUT_NEGATIVE_LABEL for l in y  ]
    positive_prf , negative_prf = calc_prf(y,ypred)
    acc = calc_acc(y,ypred) 

    print '''
    For positive class , p = {p[0]:.2%} , r = {p[1]:.2%} , f = {p[2]:.2%} 
    For negative class , p = {n[0]:.2%} , r = {n[1]:.2%} , f = {n[2]:.2%}
    Accuracy = {a:.2%}
    '''.format(p=positive_prf , n=negative_prf , a=acc)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="using xgboost for classification")
    argp.add_argument("-train" , "--train_data" , help="path to training data in libsvm format" , type=str , required=True)
    argp.add_argument('-test' , '--test_data' , help="path to testing data in libsvm format" , type=str , required=True)
    #argp.add_argument('-o' , '--output_f' , help="storing path to predict result" , type=argparse.FileType('w') , required=True)
    args = argp.parse_args()
    main(args.train_data , args.test_data)
