#!/usr/bin/env python
#coding=utf-8
import numpy as np
import os
import sys
import argparse
import logging
try :
    import cPickle as pickle
except :
    import pickle

from fileprocessing import *
logging.basicConfig(level=logging.INFO)

###########DEBUG###############

def debug_output(is_debug,des,obj) :
    if is_debug :
        print "%s%s%s" %('-'*10,des,'-'*10)
        print obj

def debug_save_SVM_data(is_debug,Y,X,dimension,basepath) :
    if is_debug :
        fpath = os.path.join(basepath , 'test.debug.svm.data')
        save_SVM_data(Y,X,dimension,fpath)
        print 'SVM data has stored at "%s"' %(fpath)

def debug_save_predict_rst(is_debug,pf,basepath) :
    if is_debug :
        fpath = os.path.join(basepath , 'predict.debug.svm.rst')
        pf_l = []
        for x in pf :
            pf_l.append(str(x[0]) + '\n')
        f = open(fpath,'w')
        f.writelines(pf_l)
        f.close()
        print 'predict result has been stored at "%s"' %(fpath)

##########END DEBUG Fn###########

def load_model(model) :
    w = pickle.load(model)
    b = pickle.load(model)
    dic = pickle.load(model)
    r = pickle.load(model)
    ngram = pickle.load(model)
    return w,b,dic,r,ngram


def NBSVM_predict(X,w,b) :
    dimension = w.shape[0]
    w.shape = (dimension , 1) # transpose
    data_size = len(X)
    #X_m = np.zeros((data_size , dimension)) #TO BIG to calc
    i = 0
    Y_pre = np.zeros((data_size , 1))
    for x in X :
        x_m = np.zeros((1,dimension))
        for idx in x :
            x_m[0,idx -1 ] = x[idx] # the idx is the the feature idx start from 1 , where matrix start from 0 . and x is a dict .
        Y_pre[i,0] = np.dot(x_m,w) + b
        i += 1
    
    #predict function : y = sign(x*w + b )
    #here X_m 's shape is (data_size,dimension) , w is (dimension,1) , b will be broadcasting to (data_size,1) , result is matrix with shape (data_size , 1)
    #Y_pre = np.dot(X_m , w ) + b ; # mul is not the star *  but the numpy.dot !!
    return np.sign(Y_pre)

def evaluation_using_liblinear(ty,py) :
    return linearutil.evaluations(ty,py)


def main(f,model,is_debug , pout_f) :
    logging.info("loading model from '%s'" %(model.name))
    w,b,dic,r,ngram = load_model(model)
    logging.info("dict dimension : %d" %(len(dic)))
    w = np.array(w)
    
    debug_output(is_debug,"w,b,dic,r,ngram",{"w":w,"b":b,"r":r,"ngram":ngram})

    logging.info('vectorize test file')
    vecs = vectorize_docs(f,dic,r,ngram)
    X = []
    for v in vecs :
        X.append({key:val for key,val in v})
    logging.info("using nbsvm predict")    
    Y_predict = NBSVM_predict(X,w,b)
    logging.info("output predict result to '%s'" %(pout_f.name))
    for y in Y_predict :
        pout_f.write("%d\n" %(y))
    pout_f.close()
    f.close()
    model.close()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="using NBSVM to predict")
    parser.add_argument("--f",help="path to file to be predicted"  , type=argparse.FileType('r') , required=True)
    parser.add_argument("--model",help="path to NBSVM model" , type=argparse.FileType('r') , required=True)
    parser.add_argument("--liblinear",help="path to liblinear",default="/users1/wxu/bin/liblinear-1.96/python")
    parser.add_argument("--DEBUG" , dest="is_debug" ,help="wheather to open the debug model" , action="store_true")
    parser.add_argument("--predict_f" , dest="pout_f" , help="path to store prediction result" , type=argparse.FileType('w') , required=True)
    args = vars(parser.parse_args())
    liblinear_path = args.pop('liblinear')
    if not os.path.exists(liblinear_path) :
        raise Exception
    sys.path.append(liblinear_path)
    import liblinearutil as linearutil
    
    main(**args)

