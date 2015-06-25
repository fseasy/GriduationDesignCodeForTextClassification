#/usr/bin/env python
#coding=utf-8

import os
import sys
import argparse
import numpy as np
import logging

from collections import Counter

from fileprocessing import unordered_ngram_tokenizer
from fileprocessing import NGRAM_JOIN_CHAR
from loadwordvector import load_wordvector
from dict_data_handler import save_dict_data

logging.basicConfig(level=logging.DEBUG)

def stat_class_info(class_f , ngram) :
    '''
    input > class_f : the trainning corpus file  of one class . Here , every line in class_f , represent a [doc] !
    return > a Counter  : ( (words : df ) ,) , means including the words and its DF num
    '''
    c = Counter()
    for line in class_f :
        doc_container = unordered_ngram_tokenizer(line , ngram)
        doc_words_set = set(doc_container)
        c.update(doc_words_set) # here update , should update the words and it's DF
    #for key , df in c.items() :
    #    print "%s %d" %(key , df)
    return c

def build_class_vector(all_dict , class_container , alpha=1) :
    '''
    input > all_dict : [ ] , list contains all words
            class_container , the container of one class , ( words and DF )
            alpha , smooth parameter , default is 1
    return > a numpy array as a class vector
    '''
    #print >> sys.stdrr ,  len(all_dict)
    v = np.zeros((1,len(all_dict)))
    #print >> sys.stderr , v.shape
    for i in range(0,len(all_dict)) :
        word = all_dict[i]
        v[0,i] = class_container[word] + alpha
        #print v[0,i]
    return v


def compute_logcount_ratio(all_dict , pos_container , neg_container , alpha=1) :
    logging.info("compute log-count ratio")
    pos_v = build_class_vector(all_dict , pos_container , alpha)
    logging.debug(pos_v)
    neg_v = build_class_vector(all_dict , neg_container , alpha)
    logging.debug(neg_v)
    pos_norm1 = np.abs(pos_v).sum()
    neg_norm1 = np.abs(neg_v).sum()
    logging.debug("pos_norm 1 : %d neg_norm 1 : %d" %(pos_norm1 , neg_norm1))
    r = np.log( (pos_v / pos_norm1) / (neg_v / neg_norm1) )
    logging.debug("logcount ration: %s" %(r))
    return r

def get_corresponding_ngrams_vector(gramlist , wordvectors ,  gram_split_char) :
    '''
    This will return updated gramlist , corresponding vectors
    input > gramlist : grams 
            wordvectors : all wordvectors
            gram_split_char : for split gram
    output > updated_gramlist : because not every gram will occurs at wordvector , so should update it !
             vectos : vectors of grams , Format is Numpy.Matrix
    '''
    logging.info("calc ngrams vector...")
    logging.info("origin dict dimension is %d" %(len(gramlist)))
    updated_gramlist = []
    vector = []
    for ngram in gramlist :
        grams = ngram.split(gram_split_char)
        ngram_vector = []
        is_found = True
        for unigram in grams :
            if unigram in wordvectors :
                ngram_vector.append(wordvectors[unigram])
            else :
                is_found = False
                break
        if not is_found : continue
        updated_gramlist.append(ngram)
        average_vector = sum(ngram_vector) / float(len(ngram_vector)) 
        vector.append(average_vector)
    logging.info("updated dict dimension is %d" %(len(updated_gramlist)))
    logging.info("calc ngrams vectors done. ")
    #for i in range(0 , len(updated_gramlist)) :
    #    print "%s %s" %(updated_gramlist[i] , vector[i])
    return updated_gramlist , vector

def main(wordvector_f , pos_f , neg_f , ngram , out_f) :
    wordvectors = load_wordvector(wordvector_f)
    pos_container = stat_class_info(pos_f , ngram)
    neg_container = stat_class_info(neg_f , ngram)
    all_dict = list(set(list(pos_container) + list(neg_container)))
    updated_dict , gram_vecs = get_corresponding_ngrams_vector(all_dict, wordvectors ,NGRAM_JOIN_CHAR)
    #print len(updated_dict)
    #print len(all_dict)
    r = compute_logcount_ratio(updated_dict , pos_container , neg_container , 1)
    save_dict_data(out_f , updated_dict , gram_vecs , r , ngram)


def query_gram(wordvectors) :
    while True :
        query = raw_input("input a word to display :")
        query = query.rstrip()
        if query == "exit" :
            break
        else :
            try :
                print wordvectors[query]
            except KeyError , e :
                print "No word '%s'" %(query)
    logging.info("exit")



if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(description="build dict from train data")
    argparser.add_argument("--postrain" , help="path to positive trainning data" , type=argparse.FileType('r') , required=True)
    argparser.add_argument("--negtrain" , help="path to negative trainning data" , type=argparse.FileType('r') , required=True)
    argparser.add_argument("--wordvector" , help="path to word vectors" , type=argparse.FileType('r') , required=True)
    argparser.add_argument("--ngram" , help="gram num" , type=int , choices=range(1,4) , default="2")
    argparser.add_argument("--out" , help="output dict data" , default=sys.stdout , type=argparse.FileType("w"))
    args = argparser.parse_args()

    main(args.wordvector , args.postrain , args.negtrain , args.ngram , args.out)

    args.wordvector.close()
    args.postrain.close()
    args.negtrain.close()
