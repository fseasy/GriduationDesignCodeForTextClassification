#/usr/bin/env python
#coding=utf-8

import sys
import argparse
import numpy as np
import logging
sys.path.append("../ready_word_vector")

logging.basicConfig(level=logging.INFO)

from dict_data_handler import load_dict_data
from dr_model_handler import save_dr_model

def get_cluster_assignment(cluster_assignment_f ) :
    assignment = []
    for line in cluster_assignment_f :
        parts = line.strip().split()
        assignment.append(int(parts[0]))
    return assignment

def generate_trans_model(gramlist , assignment , r ) :
    assert(len(gramlist) == len(assignment) == len(r))
    trans_model = {}
    for i in range(len(assignment)) :
        gram = gramlist[i]
        cluster_id = assignment[i]
        r_val = r[i]
        trans_model[gram] = (cluster_id , r_val) # using tuple , instead of dict
    return trans_model




def main(dict_data_f , cluster_assignment_f , out_f , cluster_num) :
    gramlist , v_list , r , gram_n = load_dict_data(dict_data_f)
    r = r.tolist()[0] # r is a (1,len(gramlist)) numpy.ndarray
    assignment = get_cluster_assignment(cluster_assignment_f)
    trans_model = generate_trans_model(gramlist , assignment , r)
    save_dr_model(out_f , trans_model , gram_n , cluster_num)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="ready dimensionality reduction")
    argp.add_argument("--out" , help="path to out model" , type=argparse.FileType('w') , required=True)
    argp.add_argument("--dictpath" , help="path to dict data" , type=argparse.FileType('r') , required=True)
    argp.add_argument("--cluster_assment" , help="cluster of the ngram" , type=argparse.FileType('r') , required=True)
    argp.add_argument("--cluster_num" , help="cluster num , value k in kmeans" , type=int , required=True)
    args = argp.parse_args()
    main(args.dictpath , args.cluster_assment , args.out , args.cluster_num)

    args.dictpath.close()
    args.cluster_assment.close()
    args.out.close()

