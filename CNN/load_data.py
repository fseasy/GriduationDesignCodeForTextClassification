#/usr/bin/env python
#coding=utf8

import string
import numpy
import theano
import theano.tensor as T

import traceback
# Define The global data definition

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


ALPHABET_LEN = 69
ALPHABET = string.ascii_lowercase + string.digits + "-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" # has 2 escape character '\' !

ALPHABET_VECTOR = numpy.append(numpy.zeros(( 1 , ALPHABET_LEN )) , numpy.eye(ALPHABET_LEN) , axis=0) # ( 69 + 1 ) * ( 69 )

ALPHABET_SHARED_VECTOR = theano.shared( numpy.asarray(ALPHABET_VECTOR , dtype=theano.config.floatX ) , borrow=True , name="alphabet" )

MAP_CHAR2IDX = { char:idx for char , idx in zip(ALPHABET , range(1,ALPHABET_LEN + 1))}
IGNORED_CHAR_IDX = 0

LINE_LEN_LIMIT = 1311
#LINE_LEN_LIMIT = 123



## Check definition
try:
    assert(len(ALPHABET) == ALPHABET_LEN)
    assert(ALPHABET_VECTOR.shape == ( 1 + ALPHABET_LEN , ALPHABET_LEN))
except AssertionError , e :
    print ALPHABET
    print len(ALPHABET)
    print ALPHABET_VECTOR.shape
    traceback.print_exc()
    exit(1)

def trans_chars2indices(line) :
    indices = [ ]
    for char in line :
        char = char.lower()
        if char in MAP_CHAR2IDX : indices.append(MAP_CHAR2IDX[char])
        else : indices.append(IGNORED_CHAR_IDX)
    return indices

def format_indices(indices , len_limit=LINE_LEN_LIMIT , filter_idx=IGNORED_CHAR_IDX) :
    indices.reverse() # LSTM rule
    if len(indices) >= len_limit : return indices[:len_limit]
    else :
        extra_num = len_limit - len(indices)
        for i in range(extra_num) :
            indices.append(filter_idx)
        return indices

def trans_data(f , label) :
    lines_indices = []
    labels = []
    for line in f :
        line = line.strip()
        indices = trans_chars2indices(line)
        indices = format_indices(indices)
        lines_indices.append(indices)
        labels.append( label  ) 

    return [ numpy.asarray(lines_indices) , numpy.asarray(labels) ]

if __name__ == "__main__" :
    #f_path = "/users1/wxu/code/text_category/NBSVM/data/imdb/train/postrain.imdb"
    f_path = "train.f"
    f = open(f_path)
    lines_indices , labels = trans_data(f , POSITIVE_LABEL)
    print lines_indices
    print ALPHABET_VECTOR[lines_indices.flatten()].reshape(2,1,LINE_LEN_LIMIT , ALPHABET_LEN)
    print labels
    f.close()

    
    

