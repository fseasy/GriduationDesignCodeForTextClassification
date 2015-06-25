#!/usr/bin/env python
#!coding=utf-8

import sys
import os
import logging
import traceback

def get_wrong_file(list_f_name , predict_f_name) :
    try :
        list_f = open(list_f_name)
        predict_f = open(predict_f_name)
    except IOError , e :
        logging.error("open file error : %s , %s" %(list_f_name , predict_f_name))
        logging.error(traceback.print_exc())

    error_cnt = 0 
    line_cnt = 0
    for line in list_f :
        line_cnt += 1 
        dtype_predict = predict_f.readline().strip()
        if line == "" or dtype_predict == "" :
            break
        path , dtype_real = line.strip().split()
        if dtype_real != dtype_predict :
            if error_cnt == 0 :
                print "%10s\t%-30s\t%10s\t%10s" %("line num","path" , "real type" , "predict type")
            print "%10d\t%30s\t%10s\t%10s" %(line_cnt ,path , dtype_real , dtype_predict)
            error_cnt += 1
    
    print >> sys.stderr , "total : %d predict error of %d" %(error_cnt , line_cnt)

if __name__ == "__main__" :
    if len(sys.argv) != 3 :
        logging.error("usage : %s [%s] [%s]" %(sys.argv[0] , "list_file" , "predict_file"))
        exit(0)
    get_wrong_file(sys.argv[1] , sys.argv[2])

