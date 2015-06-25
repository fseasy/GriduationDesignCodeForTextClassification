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

def compute_logcount_ratio(pos_class_counter , neg_class_counter , docs_dict , alpha=1) :
    dict_len = len(docs_dict)
    p = np.zeros(dict_len)
    q = np.zeros(dict_len)
    for word_idx in range(len(docs_dict)) :
        word = docs_dict[word_idx]
        p[word_idx] = pos_class_counter[word]
        q[word_idx] = neg_class_counter[word]  # for counter , if word no in counter , return 0
    p = p + alpha
    q = q + alpha

    p_norm1 = abs(p).sum()
    q_norm1 = abs(q).sum()

    r = np.log( (p / p_norm1) / (q / q_norm1)  )
    
    return r.tolist()

def build_SVM_format_X_from_docs_words(doc_words_list , docs_dict , r) :
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
                # if r val is 0 , don't output
                if r[idx_svm - 1] != 0 :
                    one_docs_repr[idx_svm] = r[idx_svm -1]

        X.append(one_docs_repr)
    return X

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
        py.append(np.sign( np.dot(v,w) + b) or NAGATIVE_LABEL) 
    return py



def ready_cross_validation_data(Y,X,cv_num) :
    #make a shuffle order , then split dataset by this order
    instance_num = len(X)
    shuffle_order = range(instance_num)
    np.random.shuffle(shuffle_order)
    
    if cv_num > instance_num :
        logging.warning("instance num : %d while cv_num : %d , set cv_num as instance num" %(instance_num , cv_num))
        cv_num = instance_num

    start_idx = [ i * int(instance_num / cv_num ) for i in range(cv_num ) ] 
    start_idx.append(instance_num) # for ignore the Edge condition judgement
    
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    test_order_list = []

    for ite_num in range(cv_num) :
        sub_order = []
        sub_order.extend(shuffle_order[0:start_idx[ite_num]])
        sub_order.extend(shuffle_order[start_idx[ite_num+1]:instance_num])
        
        train_X = [X[instance_id] for instance_id in sub_order]
        train_Y = [Y[instance_id] for instance_id in sub_order]
        
        train_X_list.append(train_X)
        train_Y_list.append(train_Y)
        
        sub_order = []
        sub_order.extend(shuffle_order[start_idx[ite_num]:start_idx[ite_num+1]])
        test_X = [X[instance_id] for instance_id in sub_order]
        test_Y = [Y[instance_id] for instance_id in sub_order]

        test_X_list.append(test_X)
        test_Y_list.append(test_Y)

        test_order_list.append(sub_order)
        
    return train_X_list , train_Y_list , test_X_list , test_Y_list , test_order_list

def cross_validation_grid(train_X_list , train_Y_list , test_X_list , test_Y_list , test_order_list , params) :
    w_positive_range = params["w_positive_r"]
    w_negative_range = params["w_negative_r"]
    bias_range = params['bias_r']
    c_r = params['c_r']
    beta_r = params['beta_r']
    
    cv_rst = []    
    
    cv_num = len(train_X_list)
    data_num = sum([len(l) for l in test_order_list])
    for c in c_r :
        for bias in bias_range :
            for w_i in range(len(w_positive_range)) :
                w_positive = w_positive_range[w_i]
                w_negative = w_negative_range[w_i]
                for beta in beta_range :
                    logging.info("")
                    logging.info("cross validation using C = %f , bias = %.2f , w_positive = %.2f , w_negative = %.2f , beta = %.3f\n" %(
                        c , bias , w_positive , w_negative , beta))
                    params_str = "-c %f -w%s %.2f -w%s %.2f -B %.2f -q" %( c , POSITIVE_LABEL , w_positive , NEGATIVE_LABEL , w_negative , bias  )
                    predict_Y = [ 0 for x in range(data_num)]
                    Y = [0 for x in range(data_num)]
                    for i in range(cv_num) :
                        prob = liblinearutil.problem(train_Y_list[i] , train_X_list[i])
                        param = liblinearutil.parameter(params_str)
                        model = liblinearutil.train(prob , param)
                        NB_w , NB_b = compute_NBSVM_param(model,beta)
                        #p_label , p_acc , p_val = liblinearutil.predict(test_Y_list[i] , test_X_list[i] , model , '-q')
                        p_label = NBSVM_predict(NB_w,NB_b,test_X_list[i])
                        idx = 0
                        for instance_id in test_order_list[i] :
                            predict_Y[instance_id] = p_label[idx]
                            Y[instance_id] = test_Y_list[i][idx]
                            idx += 1
                    positive_prf , negative_prf = calc_prf(Y,predict_Y)
                    positive_prf = map(lambda x:100*x , positive_prf)
                    negative_prf = map(lambda x:100*x , negative_prf)
                    print "positive class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( positive_prf[0] , positive_prf[1] , positive_prf[2])
                    print "negative class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%%" %( negative_prf[0] , negative_prf[1] , negative_prf[2])
                    sys.stdout.flush()
                    cv_rst.append({'c':c , 'w_positive':w_positive , 'w_negative':w_negative , 'bias':bias , 'beta':beta ,'positive_prf':positive_prf , 'negative_prf':negative_prf})

    certain_handle(cv_rst)
    return cv_rst


def do_cross_validation(Y,X,cv_num,params) :
    label_num = len(Y)
    instance_num = len(X)
    assert label_num == instance_num

    train_Xs , train_Ys , test_Xs , test_Ys , test_order_list = ready_cross_validation_data(Y,X,cv_num)

    return cross_validation_grid(train_Xs,train_Ys,test_Xs,test_Ys,test_order_list,params)


def certain_handle(cv_rst) :
    sorted_rst = sorted(cv_rst , key=lambda x : x['negative_prf'][2] , reverse=True)
    limit = min(3,len(sorted_rst))
    logging.info("%sFINISHED%s" %('#'*10 , '#'*10))
    print '按负例类别(minority)F值由高到低排列前%d位' %(limit)
    for i in range(limit) :
        rst = sorted_rst[i]
        print 'class : p = %6.2f %% , r = %6.2f %% , f = %6.2f%% . C = %f , bias=%.2f , w_positive = %.2f , w_negative = %.2f , beta = %.3f' %(
                rst['negative_prf'][0] , rst['negative_prf'][1] , rst['negative_prf'][2] , rst['c'] , rst['bias'] , rst['w_positive'] , 
                rst['w_negative'] , rst['beta'])



def main(pos_f , neg_f , gram_num , cv_num , rst_f , params) :
    logging.info("statistic docs info")
    pos_docs , pos_labels , pos_words_counter = stat_oneclass_docs(pos_f , POSITIVE_LABEL , gram_num)
    neg_docs , neg_labels , neg_words_counter = stat_oneclass_docs(neg_f , NEGATIVE_LABEL , gram_num)
    docs_dict = build_docs_dict([pos_words_counter , neg_words_counter])
    print "dict len" , len(docs_dict)
    # compute r
    r = compute_logcount_ratio(pos_words_counter , neg_words_counter , docs_dict)
    logging.info("ready X,Y for SVM cross validation")
    # ready X , Y
    Y = []
    Y.extend(pos_labels)
    Y.extend(neg_labels)
    
    all_docs = []
    all_docs.extend(pos_docs)
    all_docs.extend(neg_docs)
    X = build_SVM_format_X_from_docs_words(all_docs,docs_dict,r)
    
    logging.info("cross validation")
    cv_rst = do_cross_validation(Y,X,cv_num,params)
    logging.info("dump cross validation result pickle to file '%s'" %(rst_f.name))
    pickle.dump(cv_rst , rst_f)

def split_args(s) :
    return map(float , re.split(ur"[ ,]+" , s.strip("\"'")))

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="nbsvm grid search using cross validation")
    argp.add_argument("-p" , "--pos" , help="path to positive corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-n" , "--neg" , help="path to negative corpus" , required=True , type=argparse.FileType('r'))
    argp.add_argument("-g" , "--gram" , help="ngram num" , choices=[1,2,3] , required=True , type=int)
    argp.add_argument("-cv" , "--cv_num" , help="cross validation num" , required=True , type=int)
    argp.add_argument("-o" , "--rst_pickle_f" , help="gird result pickle file" , required=True , type=argparse.FileType('w'))
    ##liblinear parameter
    argp.add_argument("-c_r" , "--c_range" , help="liblinear parameter C range , like '1,2,3...'" , required=True , type=str)
    argp.add_argument("-b_r" , "--bias_range",help="liblinear parameter B range , like '1,2,3...'" , required=True , type=str)
    argp.add_argument("-w_p" , "--w_positive_range" , help="liblinear parameter wi for positive range , like 1.0,2.0,..." , required=True , type=str)
    argp.add_argument("-w_n" , "--w_negative_range" , help="liblinear parameter wi for negative range , like 1.0,2.0,..." , required=True , type=str)
    ##NBSVM parameter Beta for interpolation  
    argp.add_argument("-beta_r" , "--beta_range",help="liblinear parameter Beta range , like '0,0.1,0.2,0.3...1'" , required=True , type=str)
    
    argp.add_argument("--liblinear" , help="path liblinear python interface lib" , default="/users1/wxu/bin/liblinear-1.96/python")

    args = argp.parse_args()
    try :
        sys.path.append(args.liblinear)
        import liblinearutil
    except Exception , e :
        logging.error(e)
        exit(1)
    #param preprocess
    bias_range = split_args(args.bias_range)
    c_range = split_args(args.c_range)
    w_positive_range = split_args(args.w_positive_range)
    w_negative_range = split_args(args.w_negative_range)
    beta_range = split_args(args.beta_range)
    try :
        assert len(w_positive_range) == len(w_negative_range)
    except AssertionError , e :
        logging.error("weight for positive are not as long as negative")
        traceback.print_exc()
        exit(1)
    logging.info("%sParams set%s" %("#"*10 ,'#'*10 ))
    logging.info('bias in range %s ' %(bias_range))
    logging.info('w_positive in range %s' %(w_positive_range))
    logging.info('w_netative in range %s' %( w_negative_range))
    logging.info('C in range %s' %(c_range))
    logging.info('beta in range %s' %(beta_range))
    liblinear_param = {'bias_r':bias_range , 'c_r':c_range , 'w_positive_r':w_positive_range , 'w_negative_r':w_negative_range , 'beta_r':beta_range}
    
    main(args.pos , args.neg , args.gram , args.cv_num , args.rst_pickle_f ,liblinear_param)

    args.pos.close()
    args.neg.close()
    args.rst_pickle_f.close()
