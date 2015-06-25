#/usr/bin/env python
#coding=utf-8

try :
    import cPickle as pickle
except :
    import pickle
import numpy as np
import logging

def save_dict_data(fpo , gramlist , vector_mat , r , gram_n) :
    '''
    save the dict data to the file using pickle
    input > fpo : out file obj 
            gramlist : word dict
            vector_mat : vector , in numpy.Matrix format
            gram_n : dict contains (n) grams 
    return > True or False
    '''
    logging.info("save dict data to file '%s'" %(fpo.name))
    v_list = [ v.A[0].tolist() for v in vector_mat]
    try :
        pickle.dump(gramlist , fpo)
        pickle.dump(v_list , fpo)
        pickle.dump(r , fpo)
        pickle.dump(gram_n , fpo)
        logging.info("save successfully.")
        return True
    except Exception , e :
        logging.error(e)
        return False

def load_dict_data(fpi) :
    '''
    load dict data from pickle file
    input > fpi : pickle in file
    return > gramlist : words dict 
             vector_list : vectors , in build_in [ [] , ]
             r : log-count ratio , in numpy.array
             gram_n : int
    '''
    logging.info("load dict data from file '%s'" %(fpi.name))
    try :
        gramlist = pickle.load(fpi)
        v_list = pickle.load(fpi)
        r = pickle.load(fpi)
        gram_n = pickle.load(fpi)
        logging.info('load successfully .')
        return gramlist , v_list , r , gram_n
    except Exception , e :
        logging.error(e)
        exit(1)

