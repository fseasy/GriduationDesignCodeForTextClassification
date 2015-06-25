#/usr/bin/env python
#coding=utf-8

import logging 

try :
    import cPickle as pickle
except :
    import pickle


def save_dr_model(fpo , trans_model , gram_n , cluster_num) :
    logging.info("save model to '%s'" %(fpo.name))
    pickle.dump(trans_model , fpo)
    pickle.dump(gram_n , fpo)
    pickle.dump(cluster_num , fpo)
    logging.info("save model done .")


def load_dr_model(fpi) :
    logging.info("load model from '%s'" %(fpi.name))
    trans_model = pickle.load(fpi)
    gram_n = pickle.load(fpi)
    cluster_num = pickle.load(fpi)
    logging.info("load model done .")
    return trans_model , gram_n , cluster_num
