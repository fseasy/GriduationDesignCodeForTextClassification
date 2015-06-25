#!/usr/bin/env python
#coding=utf-8

import sys
import os
import logging
import random
import argparse
import numpy as np

from fileprocessing import *

logging.basicConfig(level=logging.INFO)

def abstract_words_for_every_doc(f_obj , ngram) :
    '''
    intput > f_obj : file object , every line stands for a doc
             ngram : 2 or 1
    return > [ [],[]  ] , enery element is the doc's words , DO NOT remove the dulplicate words!!
    '''
    rst = []
    for l in f_obj.xreadlines() :
        doc_words = tokenize(l,ngram)
        rst.append(doc_words)
    return rst

def build_doc_info(labels , docs_l) :
    '''
    input > labels : list of lables
            docs_l : [ [] , [] ...] , every element stands a labeled docs corresponding to the label of `labels` , and the docs is a list of doc represented by words list , with dulplicate words 
    return > list [ {} , ...] , every element like {label , [  ] ,doc_id } , where label is the doc label , and list in it is the words represent the doc , and has no redundancy , that means , no dulplicate words ; store doc_id for verifying the predict result
    '''
    doc_info = []
    doc_id = 1 
    for label , docs in zip(labels , docs_l) :
        for doc in docs :
            doc = list(set(doc))
            doc_info.append({'label':label , 'doc':doc , 'doc_id': '_'.join([str(label),str(doc_id)]) })
            doc_id += 1
    return doc_info

def split_docs_to_k_folds(k,doc_info) :
    '''
    return > doc_folds : splited doc info
             has_splited : bool
    '''
    num = len(doc_info)
    n_per_fold = int(num / k ) #floor
    has_splited = ( n_per_fold != 0 ) # that means , if num < k , can not be splited
    doc_folds = []
    for i in range(0,k-1) :
        doc_folds.append(doc_info[i*n_per_fold:(i+1)*n_per_fold])
    doc_folds.append(doc_info[(k-1)*n_per_fold:]) #put left to the last flod
    return doc_folds , has_splited

def build_dict_for_k_folds(doc_folds , exclude_idx) :
    dic = {}
    idx = 1
    for i in range(len(doc_folds)) :
        if i == exclude_idx :
            continue
        docs = doc_folds[i]
        for doc_dict in docs : # [ {'label': , 'doc':[] , 'doc_id ': } ]
            for word in doc_dict['doc'] :
                if word not in dic :
                    dic[word] = idx 
                    idx += 1
    return dic

def build_docs_matrix(doc_folds , test_part_idx , total_docs_num , dic) :
    '''
    input > doc_folds : list for doc folds , [ [] ] element is list , and in the list is the doc_dict , {'label':' ' , 'doc':[]}
            test_part_idx : the fold id for test
            total_docs_num : the total number of docs 
            dic : the dic of feature 
    return < matrix for train_data , test_data, train_label , test_label
    '''
    test_docs_num = len(doc_folds[test_part_idx])
    train_docs_num = total_docs_num - test_docs_num 
    dimension = len(dic)
    train_data_matrix = np.zeros((train_docs_num , dimension))
    test_data_matrix = np.zeros((test_docs_num , dimension))
    
    train_label_matrix = np.zeros((train_docs_num , 1))
    test_label_matrix = np.zeros((test_docs_num , 1))
    #build it
    #test
    for doc_id in range(0,test_docs_num) :
        doc = doc_folds[test_part_idx][doc_id]['doc']
        for word in doc :
            if word in dic :
                word_id_SVM = dic[word]
                test_data_matrix[doc_id,word_id_SVM - 1] = 1 # bool
        test_label_matrix[doc_id,0] = doc_folds[test_part_idx][doc_id]['label']
    #train
    train_doc_id = 0
    for i in range(0,len(doc_folds)) :
        if i == test_part_idx :
            continue
        docs = doc_folds[i]
        for doc_dict in docs :
            doc_label = doc_dict['label']
            train_label_matrix[train_doc_id,0] = doc_label
            doc = doc_dict['doc']
            for word in doc :
                word_id_SVM = dic[word]
                train_data_matrix[train_doc_id,word_id_SVM - 1] = 1
            train_doc_id += 1
    return train_data_matrix , test_data_matrix , train_label_matrix , test_label_matrix


def compute_logcount_ratio(train_matrix , train_label, alpha=1) :
    data_size , dimension = train_matrix.shape
    p = np.zeros((1,dimension)) + alpha 
    q = np.zeros((1,dimension)) + alpha
    for i in range(data_size) :
        if train_label[i,0] == POSITIVE_LABEL :
            p += train_matrix[i]
        else :
            q += train_matrix[i]
    p_norm1 = abs(p).sum()
    q_norm1 = abs(q).sum()
    r = np.log( (p / p_norm1 ) / (q / q_norm1) )
    return r

def build_docs_matrix_for_NBSVM(r , train_matrix , test_matrix) :
    '''
    input > r : log-count ratio
            train_matrix , test_matrix : original matrix of boolean value
    return > train_matrix , test_matrix for NBSVM 
    '''
    train_matrix = train_matrix * r
    test_matrix = test_matrix * r
    return train_matrix , test_matrix 

def trans_nparray_to_SVM_format(label,data) :
    Y = []
    X = []
    Y = [ l for l in label[:,0]]
    for row in data :
        #X.append(list(row))
        row = list(row)
        doc_v = {}
        for idx in range(len(row)) :
            if row[idx] != 0 :
                doc_v[idx+1] = row[idx]
        X.append(doc_v)
    return Y , X

def trans_nparray_label_to_SVM_format(label) :
    return [l for l in label[:,0]]


def train_using_liblinear(Y,X,options) :
    m = linearutil.train(Y,X,options)
    return m

def compute_NBSVM_param(m , beta=0.25) :
    '''
    copy from nbsvm-train.py
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

def NBSVM_predict(w,b,test_data) :
    w = np.array(w)
    w.shape = (w.shape[0],1)
    return np.sign( np.dot(test_data,w) + b) # data_size * 1
    
    
def evaluation_using_liblinear(ty,py) :
    return linearutil.evaluations(ty,py)

def k_fold_cross_validation(k,doc_info) :
    '''
    intput > k : cross validation k-flods 
             doc_info : shuffled document info produced by `build_doc_info`
    return > none
    '''
    total_doc_num = len(doc_info)
    doc_folds , has_splited  = split_docs_to_k_folds(k,doc_info)
    if not has_splited :
        logging.error("data set is too small to using %d fold validation" %(k))
        return
    P_Y = [] # predict labels
    T_Y = [] # true labels
    for i in range(0,k) :
        # start k floads , select the i slice as the test , other as to be training data
        logging.info("----fold %d----" %(i+1))
        dic = build_dict_for_k_folds(doc_folds , i)
        logging.info("build docs matrix")
        train_data, test_data , train_label , test_label = build_docs_matrix(doc_folds,i,total_doc_num,dic)
        logging.info("compute logcount-ratio")
        r = compute_logcount_ratio(train_data,train_label)
        logging.info("build docs matrix for SVM : reset the feature value using r")
        train_data , test_data = build_docs_matrix_for_NBSVM(r,train_data,test_data)
        logging.info("trans docs matrix to libSVM format")
        Y , X = trans_nparray_to_SVM_format(train_label , train_data)
        logging.info("training using liblinear")
        m = train_using_liblinear(Y,X, "-c 1 -s 1")
        logging.info("compute NBSVM param")
        w,b = compute_NBSVM_param(m)
        logging.info("NBSVM predict")
        p_y = NBSVM_predict(w,b,test_data)        
        p_y = trans_nparray_label_to_SVM_format(p_y)
        Y = trans_nparray_label_to_SVM_format(test_label)
        logging.info("recode the predict result labels true labels")
        T_Y.extend(Y)
        P_Y.extend(p_y)
    #over
    ACC , MSE , SCC = evaluation_using_liblinear(T_Y,P_Y)
    print "\n\nACC=%.2f%% MSE=%.2f SCC=%.2f" %(ACC,MSE,SCC)
    return T_Y , P_Y 

def output_the_predict_detail(doc_id_l , ty_l , py_l , f_outf) :
    assert(len(doc_id_l) == len(ty_l) == len(py_l) )
    for i in range(len(doc_id_l)) :
        f_outf.write("%10s\t%10d\t%10d\n" %(doc_id_l[i] , ty_l[i] , py_l[i]))


def main(pos , neg , ngram , cv_k , p_outf) :
    logging.info("abstract words")
    pos_doc_words = abstract_words_for_every_doc(pos , ngram)
    neg_doc_words = abstract_words_for_every_doc(neg , ngram)
    doc_info = build_doc_info([POSITIVE_LABEL , NEGATIVE_LABEL] , [pos_doc_words , neg_doc_words])
    random.shuffle(doc_info)
    T_Y , P_Y = k_fold_cross_validation(cv_k , doc_info)
    output_the_predict_detail([x['doc_id'] for x in doc_info] , T_Y , P_Y , p_outf)
    p_outf.close()
    pos.close()
    neg.close()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="nbsvm cross validation")
    parser.add_argument("--pos" , default="data/pos" , help="path to positive data" , type=argparse.FileType('r'))
    parser.add_argument("--neg" , default="data/neg" , help="path to negative data" , type=argparse.FileType('r'))
    parser.add_argument("--ngram" , default="2" , choices=[1,2] , help="path to negtive train data" , type=int)
    parser.add_argument("--CV" ,dest="cv_k",default="10" , help="cross validation flods" , choices=range(2,15),type=int)
    parser.add_argument("--liblinear" , default="/users1/wxu/bin/liblinear-1.96/python" , type=str)
    parser.add_argument("--pout",dest="p_outf" , default="cv_predict.rst" , help="path to cross validation predict result" , type=argparse.FileType('w'))
    args = vars(parser.parse_args())
    liblinear_path = args.pop('liblinear')
    try :
        sys.path.append(liblinear_path)
        import liblinearutil as linearutil
    except Exception , e :
        logging.error("faild to find the liblinear at '%s'" %liblinear_path)
        logging.error(e)
        exit(0)

    main(**args)
    
