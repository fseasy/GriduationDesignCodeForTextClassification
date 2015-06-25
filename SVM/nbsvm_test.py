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


def main(postest,negtest,model,is_debug , pout_f) :
    logging.info("loading model from '%s'" %(model.name))
    w,b,dic,r,ngram = load_model(model)
    logging.info("dict dimension : %d" %(len(dic)))
    w = np.array(w)
    
    debug_output(is_debug,"w,b,dic,r,ngram",{"w":w,"b":b,"r":r,"ngram":ngram})

    logging.info('vectorize test file')
    pos_vec = vectorize_docs(postest,dic,r,ngram)
    neg_vec = vectorize_docs(negtest,dic,r,ngram)
    Y , X = ready_SVM_data([POSITIVE_LABEL , NEGATIVE_LABEL] , [pos_vec , neg_vec]) 
    
    debug_save_SVM_data(is_debug,Y,X,len(dic),os.path.split(postest.name)[0])

    Y_predict = NBSVM_predict(X,w,b)
    
    debug_save_predict_rst(is_debug,Y_predict,os.path.split(postest.name)[0])

    ACC , MSE , SCC = evaluation_using_liblinear(Y,Y_predict)
    
    print "ACC:%f\nMSE:%f\nSCC:%f" %(ACC , MSE , SCC)
    doc_ids = []
    idx = 1
    for idx in range(len(pos_vec)) :
        doc_ids.append("_".join([str(POSITIVE_LABEL) , str(idx)]))
        idx += 1
    idx = 1
    for idx in range(len(neg_vec)) :
        doc_ids.append("_".join([str(NEGATIVE_LABEL) , str(idx)]))
        idx += 1
    output_predict_detail(doc_ids , Y, Y_predict , pout_f )
    
    pout_f.close()
    postest.close()
    negtest.close()
    model.close()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="using NBSVM to predict")
    parser.add_argument("--postest",help="path to positive test file" , default="data/postest" , type=argparse.FileType('r'))
    parser.add_argument("--negtest",help="path to negative test file" , default="data/negtest" , type=argparse.FileType('r'))
    parser.add_argument("--model",help="path to NBSVM model",default="out.model" , type=argparse.FileType('r'))
    parser.add_argument("--liblinear",help="path to liblinear",default="/users1/wxu/bin/liblinear-1.96/python")
    parser.add_argument("--DEBUG" , dest="is_debug" ,help="wheather to open the debug model" , action="store_true")
    parser.add_argument("--predict_f" , dest="pout_f" , help="path to store prediction result" , type=argparse.FileType('w') , default="predict.result")
    args = vars(parser.parse_args())
    liblinear_path = args.pop('liblinear')
    if not os.path.exists(liblinear_path) :
        raise Exception
    sys.path.append(liblinear_path)
    import liblinearutil as linearutil
    
    main(**args)

