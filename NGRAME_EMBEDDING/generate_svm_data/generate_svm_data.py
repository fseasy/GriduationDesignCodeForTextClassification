#/usr/bin/env python
#coding=utf-8

import sys
import argparse
import numpy as np
import logging
from collections import Counter

sys.path.append("../ready_word_vector")

from dr_model_handler import load_dr_model
from fileprocessing import unordered_ngram_tokenizer

logging.basicConfig(level=logging.INFO)


def generate_data_in_svm_format(tokens_list , trans_model , data_dimension , label) :
    vec = [0.0,]*data_dimension
    for token in tokens_list :
        if token in trans_model :
            idx = trans_model[token][0]
            r_val = trans_model[token][1]
            if abs(vec[idx]) < abs(r_val) :
                vec[idx] = r_val
    data = " ".join([ ":".join(map(str , [key+1 , val])) for key , val in enumerate(vec) ])
    return "%s %s\n" %(str(label) , data)


def main(raw_data_f , data_label , drmodel_f , out_f) :
    trans_model , gram_n , cluster_num = load_dr_model(drmodel_f)
    logging.info("gram_n : %d " %(gram_n ))
    logging.info("cluster_num : %d" %(cluster_num))
    for line in raw_data_f :
        c = unordered_ngram_tokenizer(line , gram_n)
        svm_data = generate_data_in_svm_format(c.keys() , trans_model , cluster_num , data_label)
        out_f.write(svm_data)


if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="generate svm data in low dimention")
    argp.add_argument("--rawdata" , help="raw data of documents" , type=argparse.FileType('r') , required=True)
    argp.add_argument("--label" , help="data label" , type=str , required=True)
    argp.add_argument("--out" , help="result output file" , type=argparse.FileType('w') , required=True)
    argp.add_argument("--drmodel" , help="path to dimensionaliry reduction model" , type=argparse.FileType('r') , required=True)

    args = argp.parse_args()
    main(args.rawdata , args.label , args.drmodel , args.out)

    args.rawdata.close()
    args.out.close()
    args.drmodel.close()
