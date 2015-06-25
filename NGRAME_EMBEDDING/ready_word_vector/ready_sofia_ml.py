#/usr/bin/env python
#coding=utf-8

import sys
import argparse
import logging

import numpy as np
from dict_data_handler import load_dict_data

logging.basicConfig(level=logging.INFO)

def output_vlist_in_svm_format(v_list , out_f) :
    label = 1
    for v in v_list :
        data = " ".join([ ':'.join(map(str , [idx + 1 , val])) for idx , val in enumerate(v) ])
        line = "%d %s\n" %(label , data)
        out_f.write(line)

def main(dict_data_f , out_f) :
    logging.info("load dict data from '%s'" %(dict_data_f.name))
    gramlist , v_list , r , gram_n = load_dict_data(dict_data_f)
    logging.info("load dict data done .")
    logging.info("save dict data in svm format ...")
    output_vlist_in_svm_format(v_list,out_f)
    logging.info("done. saved at '%s'" %(out_f.name))


if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="ready ngram vector in svm-light format")
    argp.add_argument("--dictpath" , help="path to dict data pickle" , type=argparse.FileType('r') , required=True)
    argp.add_argument("--out" , help="path to store the result" , type=argparse.FileType('w') , required=True)

    args = argp.parse_args()

    main(args.dictpath , args.out)

    args.dictpath.close()
    args.out.close()
