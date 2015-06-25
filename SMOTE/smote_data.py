#/usr/bin/env python
#coding=utf8

import sys
import os
import argparse
import logging
import numpy
from collections import Counter

cur_dir_path= os.path.split(os.path.abspath(__file__))[0]
add_path= os.path.join(cur_dir_path , "../BOOSTEDTREE")
sys.path.append(add_path)

from file_handler import load_for_libsvm_sparse_format as load_data , save_in_libsvm_sparse_format as save_data
'''
X,Y load_data(ifi)
    save_data(Y,X,ofi)
'''

logging.basicConfig(level=logging.INFO)


def select_data(X,Y,selected_label) :
    s_Y = []
    s_X = []
    assert len(X) == len(Y)
    for i in range(len(Y)) :
        if Y[i] == selected_label :
            s_Y.append(Y[i])
            s_X.append(X[i])
    return s_X , s_Y

def calc_Euclidean_distance(X) :
    size = len(X)
    adjacent_matrix = numpy.zeros((size , size))
    for i in range(size) :
        for j in range(i+1 , size) :
            x1 , x2 = X[i] , X[j]
            distance_sqr = 0
            all_keys = []
            all_keys.extend(x1.keys())
            all_keys.extend(x2.keys())
            all_keys = set(all_keys)
            for key in all_keys :
                val1 = 0 if key not in x1 else x1[key]
                val2 = 0 if key not in x2 else x2[key]
                distance_sqr += (val1 - val2)**2
            adjacent_matrix[(i,j),(j,i)] = distance_sqr
    adjacent_matrix[adjacent_matrix==0] = numpy.inf # if distance num is zero , in fact it should be set to inf !
    return adjacent_matrix

def generate_new_instance(x_ori , x_nei , interp) :
    all_keys = set(x_ori.keys() + x_nei.keys())
    new_ins = {}
    for key in all_keys :
        val1 = 0 if key not in x_ori else x_ori[key]
        val2 = 0 if key not in x_nei else x_nei[key]
        new_val = val2 +  (val1 - val2 ) * interp 
        new_ins[key] = new_val
    return new_ins

def smote(X , label , ratio , adj_matrix) :
    '''
    :param X : origin x insrances
    :param label : label of X
    :param ratio : ratio of smote_num / origin_num
    :param adj_matrix : adjacent matrix
    
    :return smote_X :  new X by smote not included origin X
    :return smote_Y :  new Y
    '''
    smote_X = []
    smote_Y = []
    k = min(ratio *2  , len(X) )
    for i in range(len(X)) :
        x_ori = X[i]
        nearest_neighbors_idx = adj_matrix[i,:].argsort()[0:k]
        for i in range(ratio) : # every one generate ratio num smote instance
            idx = nearest_neighbors_idx[numpy.random.randint(k)]
            neighbor_ins = X[idx]
            interpolation = numpy.random.uniform()
            new_ins = generate_new_instance(x_ori , neighbor_ins , interpolation)
            smote_X.append(new_ins)
            smote_Y.append(label)
    return smote_X , smote_Y
            
def main(ifi , ofi , selected_label , ratio) :
    logging.info('loading data ...')
    X , Y = load_data(ifi)
    instance_num = len(Y)
    labels = Counter(Y)
    try :
        assert selected_label in labels
    except AssertionError , e :
        print 'Not valid label of %d' %(selected_label)
        print 'valid labels should be %s' %(labels.keys())
        return -1
    logging.info('done. instance num %d , info %s' %(instance_num , labels))
    
    logging.info('selecting data')
    s_X , s_Y = select_data(X,Y,selected_label)
    selected_num = len(s_Y)
    logging.info('done. selected num %d' %(selected_num))

    logging.info('calc Euclidean distance')
    adjacent_matrix = calc_Euclidean_distance(s_X)
    logging.info('done . sample : ')
    print >>sys.stderr , adjacent_matrix

    logging.info('smote data')
    smote_X , smote_Y = smote(s_X , selected_label , ratio , adjacent_matrix)
    logging.info('done. generate new instance %d' %(len(smote_Y)))

    X.extend(smote_X)
    Y.extend(smote_Y)
    logging.info('smote done. now total instance %d' %(len(Y)))

    logging.info('saving data')
    save_data(Y,X,ofi)
    logging.info('done')


if __name__ == '__main__' :
    argp = argparse.ArgumentParser(description="smote data")
    argp.add_argument('-d' , help='path to origin data of sparse libsvm format' , type=argparse.FileType('r') , required=True)
    argp.add_argument('-o' , help='path to save data' , type=argparse.FileType('w') , required=True)
    argp.add_argument('-s' , help='selected label for smote' , type=int , required=True)
    argp.add_argument('-r' , '--ratio' , help="ratio for smote data to ori selectecd data" , type=int , required=True)
    args = argp.parse_args()

    main(args.d , args.o , args.s , args.ratio)

    args.d.close()
    args.o.close()

