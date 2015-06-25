#!/usr/bin/env python
#coding=utf-8

import os
import sys
import logging
import traceback

logging.basicConfig(level=logging.INFO)

def build_libsvm_data(feature_s , data_set_dir , target_path) :
    if not ( os.path.exists(data_set_dir) and os.path.isdir(data_set_dir)) :
        logging.error("'%s' is not a valid dir" %(data_set_dir))
    try :
        target_f = open(target_path , "w")
    except IOError , e :
        logging.error("failed to open file '%s'" %(target_path))
        logging.error(traceback.print_exc())
        return

    file_list = os.listdir(data_set_dir)
    dtype_trans = {"t1" : 1 , "t2" : 2}
    feature_dict = dict.fromkeys(feature_s["word"])
    feature_words = feature_s["word"]
    feature_idf = feature_s["idf"]
    
    #need to output the doc and the type sequencely
    out_list_f_name = os.path.splitext(target_path)[0] + ".list"
    out_list_f = open(out_list_f_name , "w")
    for file_name in file_list :
        path = os.path.join(data_set_dir , file_name)
        dtype = file_name.split('_')[0]
        assert dtype in dtype_trans
        dtype_numeric = dtype_trans[dtype]
        #output the list
        out_list_f.write("%s\t%d\n" %(path , dtype_numeric))
        tf_dict = {}
        #stat tf for feature
        read_file = open(path)
        for line in read_file :
            words = line.strip().split()
            for word in words :
                if word in feature_dict :
                    if word not in tf_dict :
                        tf_dict[word] = 1
                    else :
                        tf_dict[word] += 1
        read_file.close()
        #calc the normalize factor
        tf_mul_idf = []
        sum_val = 0 
        logging.debug("doc : %s\n" %(file_name))
        for i in range(len(feature_words)) :
            f = feature_words[i]
            tf = 0
            if f in tf_dict :
                tf = tf_dict[f]
            idf = feature_idf[i]
            mul_val = tf * idf ;
            tf_mul_idf.append(mul_val)
            sum_val += mul_val**2
            logging.debug("%s,%f,%f,%f\n" %(f , tf , idf , mul_val))
        normalize_num = sum_val**.5
        logging.debug("normalize_num : %f\n" %(normalize_num))
        if normalize_num == 0 :
            logging.error("Empty vector for doc '%s'" %(file_name))
            normalize_num = 1.0
        tf_idf_norm = [x / normalize_num for x in tf_mul_idf]
        #write
        target_f.write("%d" %(dtype_numeric)) 
        for i in range(len(feature_words)) :
            target_f.write(" %d:%f" %(1+i , tf_idf_norm[i]))
        target_f.write("\n")
    target_f.close()
    out_list_f.close()

def load_feature_s(feature_s , path) :
    if not os.path.exists(path) :
        return False
    read_file = open(path)
    feature_s["idf"] = []
    feature_s["word"] = []
    for line in read_file :
        line = line.strip()
        parts = line.split()
        word = parts[0]
        idf = float(parts[2])
        feature_s["idf"].append(idf)
        feature_s["word"].append(word)
    read_file.close()
    logging.info("feature loaded!")
    return True

if __name__ == "__main__" :
    if len(sys.argv) != 4 :
        logging.error("usage : %s [%s] [%s] [%s]" %(sys.argv[0] , "feature_info_path" , "raw_data_path" , "target_path"))
        exit(0)
    feature_s = {}
    load_state = load_feature_s(feature_s , sys.argv[1])
    if load_state :
        build_libsvm_data(feature_s , sys.argv[2] ,sys.argv[3])

