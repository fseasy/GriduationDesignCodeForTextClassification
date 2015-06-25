#!/usr/bin/env python
#coding=utf-8

import os
import sys
import logging 
import argparse

import nbsvm_train as nbt
import nbsvm_predict as nbp
import fileprocessing

def vectorize_docs_using_bool_feature(f_obj , dic , ngram) :
    f_vecs = []
    for l in f_obj :
        tokens = fileprocessing.tokenize(l,ngram)
        tokens = list(set(tokens))
        index = []
        for word in tokens :
            if word in dic :
                index.append(dic[word])
        index.sort()
        line_vector = []
        for idx in index :
            line_vector.append((idx , 1))
        f_vecs.append(line_vector)

    return f_vecs

def train_test_mode(postrain,negtrain,postest,negtest,ngram , params_str,pout_f) :
    logging.info("counting")
    pos_con = nbt.counting(postrain , ngram)
    neg_con = nbt.counting(negtrain , ngram)
    logging.info("abstract features")
    dic = nbt.abstract_features(pos_con , neg_con)
    
    logging.info("generate training data in libSVM format")
    postrain.seek(0,os.SEEK_SET) # the file has been read to the end , so move to the head
    negtrain.seek(0,os.SEEK_SET)
    pos_f_vecs = vectorize_docs_using_bool_feature(postrain , dic , ngram)
    neg_f_vecs = vectorize_docs_using_bool_feature(negtrain , dic , ngram)
    Y , X = fileprocessing.ready_SVM_data([fileprocessing.POSITIVE_LABEL,fileprocessing.NEGATIVE_LABEL],[pos_f_vecs,neg_f_vecs])
    

    logging.info("training using liblinear ")
    m = linearutil.train(Y,X,params_str) # -s 1 : using L2-resularized L2-loss SVM ;
    
    logging.info("generate testing data")
    pos_t = vectorize_docs_using_bool_feature(postest , dic , ngram)
    neg_t = vectorize_docs_using_bool_feature(negtest , dic , ngram)
    
    Y,X = fileprocessing.ready_SVM_data([fileprocessing.POSITIVE_LABEL ,fileprocessing.NEGATIVE_LABEL],[pos_t , neg_t])

    p_labels , p_acc , p_vals = linearutil.predict(Y,X,m)
    
    print "ACC:%.2f%% MSE:%.2f SCC:%.2f" %(p_acc[0],p_acc[1],p_acc[2])
    
    doc_ids = []
    idx = 1
    for idx in range(len(pos_t)) :
        doc_ids.append("_".join([str(fileprocessing.POSITIVE_LABEL) , str(idx)]))
        idx += 1
    idx = 1
    for idx in range(len(neg_t)) :
        doc_ids.append("_".join([str(fileprocessing.NEGATIVE_LABEL) , str(idx)]))
        idx += 1
    fileprocessing.output_predict_detail(doc_ids , Y , p_labels , pout_f )

    pout_f.close()
    postrain.close()
    negtrain.close()
    postest.close()
    negtest.close()

def cv_mode(pos,neg,ngram,cv_num , params_str) :
    logging.info("counting")
    pos_con = nbt.counting(pos , ngram)
    neg_con = nbt.counting(neg , ngram)
    logging.info("abstract features")
    dic = nbt.abstract_features(pos_con , neg_con)
    
    logging.info("generate data in libSVM format")
    pos.seek(0,os.SEEK_SET) # the file has been read to the end , so move to the head
    neg.seek(0,os.SEEK_SET)
    pos_f_vecs = vectorize_docs_using_bool_feature(pos , dic , ngram)
    neg_f_vecs = vectorize_docs_using_bool_feature(neg , dic , ngram)
    Y , X = fileprocessing.ready_SVM_data([fileprocessing.POSITIVE_LABEL,fileprocessing.NEGATIVE_LABEL],[pos_f_vecs,neg_f_vecs])    

    logging.info("%d-fold cross validing using liblinear " %(cv_num))
    m = linearutil.train(Y,X,params_str + " -v " + str(cv_num)) # -s 1 : using L2-resularized L2-loss SVM ; -c 0.1
    
    pos.close()
    neg.close()


def main(cv_num,postrain,negtrain,postest,negtest,pos,neg,ngram,params,pout_f) :
    '''
    input > postrain,negtrain,out : file object for positive train data , negative train data and output model path
            gram ; int value , 1 or 2 , decide using unigram or both unigram and bigram
    '''
    c = params['c']
    w_p = params['w_p']
    w_n = params['w_n']
    b = params['b']
    params_str = "-s 1 -c %f -B %f -w%s %f -w%s %f" %(c , b , fileprocessing.POSITIVE_LABEL , w_p , fileprocessing.NEGATIVE_LABEL , w_n)
    
    if cv_num == 1 :
        try :
            postrain = open(postrain)
            negtrain = open(negtrain)
            postest = open(postest)
            negtest = open(negtest)
            train_test_mode(postrain,negtrain,postest,negtest,ngram ,params_str, pout_f)
        except IOError , e :
            logging.error(e)
            return 0
    else :
        try :
            pos = open(pos)
            neg = open(neg)
            cv_mode(pos,neg,ngram,cv_num,params_str)
        except IOError , e :
            logging.error(e)
            return 0

    logging.info("FINISHED")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Run SVM test for contrast")
    parser.add_argument('--CV',dest="cv_num" ,help="if cv = 1 , using train / test mode , others in range(2,15) using cross validation" , 
                        choices=range(1,16),default="1",type=int)
    parser.add_argument('--postrain',help="path to positive train data",type=str)
    parser.add_argument('--negtrain',help="path to negtive train data",type=str)
    parser.add_argument('--postest',help="path to negtive train data",type=str)
    parser.add_argument('--negtest',help="path to negtive train data",type=str)
    parser.add_argument('--pos' , help="path to positive data , CV mode only")
    parser.add_argument('--neg' , help="path to negative data , CV mode only")
    parser.add_argument('--ngram',help="1 or 2 to decide using the unigram or bigram",type=int,default="2",choices=[1,2])
    parser.add_argument('--liblinear',help="path to liblinear" , default="/users1/wxu/bin/liblinear-1.96/python")
    parser.add_argument("--predict_f" , dest="pout_f" , help="path to store prediction result" , type=argparse.FileType('w') , default="svm_predict.result")
    
    parser.add_argument("-c" , "--c" , help="liblinear parameter C " , default="1" , type=float)
    parser.add_argument("-b" , "--bias",help="liblinear parameter B " , default="-1" , type=float)
    parser.add_argument("-w_p" , "--w_positive" , help="liblinear parameter wi for positive" , default="1" , type=float)
    parser.add_argument("-w_n" , "--w_negative" , help="liblinear parameter wi for negative" , default="1" , type=float)
    
    args = vars(parser.parse_args())
    if not os.path.exists(args['liblinear']) :
        logging.error("please speciafy the libLinear path")
        raise Exception
    sys.path.append(args['liblinear'])
    args.pop('liblinear') # the follow does't need it any more , delete it for a call
    import liblinearutil as linearutil # it may be not good  - -


    params = {'c' : args['c'] , 'b':args['bias'] , 'w_p' : args['w_positive'] , 'w_n':args['w_negative'] }
    main(args['cv_num'] , args['postrain'] , args['negtrain'] , args['postest'] , args['negtest'] , args['pos'] , args['neg'] , args['ngram'] , params , args['pout_f'])
