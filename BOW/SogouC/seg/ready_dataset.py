#!/usr/bin/env python
#coding=utf-8

#   For this raw data , having 9 types placed at : C000008 , C000010 , C000013 , C000014 , C000016 , C000020 , C000022 , C000023 , C000024
#   we choose the C000014 as the positive , while C000013 , C000016 , C000020 , C000022 as the negative .
#   all text num in C000014 is 1990 .
#       so the others we choose 1990 / 4 = 497 with 2 more .
#       so C000013 -> 498
#          C000016 -> 498
#          C000020 -> 497
#          C000022 -> 497
#   
#   and the train set and test set size ratio was set to -> 4 : 1
#   
#   under this set :
#   ALL : 3980
#
#   train set :
#        C000014 : 1592
#        C000013 : 396
#        C000016 : 396
#        C000020 : 396
#        C000022 : 396
#       TOTAL : 3176
#   test set :
#        C000014 : 398
#        C000013 : 102
#        C000016 : 102
#        C000020 : 101
#        C000022 : 101
#       TOTAL : 804
#
#   File name info : 10.txt ~ 1999.txt
#copy opration

import os
import sys
import logging

C14_TRAIN_NUM = 1592
C14_TEST_NUM = 398

C13_TRAIN_NUM = 396
C13_TEST_NUM = 102

C16_TRAIN_NUM = 396
C16_TEST_NUM = 102

C20_TRAIN_NUM = 396
C20_TEST_NUM = 101

C22_TRAIN_NUM = 396
C22_TEST_NUM = 101

POSITIVE_TYPE_NAME = "t1"
NEGATIVE_TYPE_NAME = "t2"

def move(source_dir_name , target_dir_name , source_idx_start , target_idx_start , num) :
    out_idx = target_idx_start
    for cnt in range(source_idx_start , source_idx_start + num ) :
        in_file_name = str(cnt) + '.txt'
        source_path = os.path.join(source_dir_name , in_file_name)
        if not os.path.exists(source_path) :
            logging.error("no file at '%s'" , source_path)
            return
        out_file_name = str(out_idx) + '.txt'
        out_idx += 1
        out_path = os.path.join(target_dir_name , out_file_name)
        if os.path.exists(out_path) :
            logging.error("out file '%s' has already exists"  %(out_path))
            logging.error("target_idx_start : %s" %(target_idx_start))
            return 
        cmd = " ".join(["cp" , source_path , out_path])
        #os.system(cmd)
        #print >> sys.stderr , cmd
def set_type(target_dir_name , start_idx , num , type_name) :
    if not os.path.exists(target_dir_name) :
        return
    for cnt in range(start_idx , start_idx + num) :
        file_name = str(cnt) + '.txt'
        source_path = os.path.join(target_dir_name , file_name)
        target_path = os.path.join(target_dir_name , type_name + '_' +  file_name)
        cmd = " ".join(["mv" , source_path , target_path])
        os.system(cmd)
        print >> sys.stderr , cmd


def ready_data_set(base_dir , target_dir) :
    target_training_set_dir = os.path.join(target_dir , "trainning_set")
    target_test_set_dir = os.path.join(target_dir , "test_set")
    if not os.path.exists(target_training_set_dir) :
        os.makedirs(target_training_set_dir)
    if not os.path.exists(target_test_set_dir) :
        os.makedirs(target_test_set_dir)
    training_out_idx = 1
    test_out_idx = 1
    #COPY 
    #C000014
    dir_path = os.path.join(base_dir , "C000014")
    if not os.path.exists(dir_path) :
        logging.error("No dir named '%s'" %(dir_path))
        return 
    move(dir_path , target_training_set_dir , 10 , training_out_idx , C14_TRAIN_NUM)
    move(dir_path , target_test_set_dir , 10 + C14_TRAIN_NUM , test_out_idx , C14_TEST_NUM)
    training_out_idx += C14_TRAIN_NUM
    test_out_idx += C14_TEST_NUM

    dir_path = os.path.join(base_dir , "C000013")
    move(dir_path , target_training_set_dir , 10 , training_out_idx , C13_TRAIN_NUM)
    move(dir_path , target_test_set_dir , 10 + C13_TRAIN_NUM , test_out_idx , C13_TEST_NUM)
    training_out_idx += C13_TRAIN_NUM
    test_out_idx += C13_TEST_NUM
    
    dir_path = os.path.join(base_dir , "C000016")
    move(dir_path , target_training_set_dir , 10 , training_out_idx , C16_TRAIN_NUM)
    move(dir_path , target_test_set_dir , 10 + C16_TRAIN_NUM , test_out_idx , C16_TEST_NUM)
    training_out_idx += C16_TRAIN_NUM
    test_out_idx += C16_TEST_NUM

    dir_path = os.path.join(base_dir , "C000020")
    move(dir_path , target_training_set_dir , 10 , training_out_idx , C20_TRAIN_NUM)
    move(dir_path , target_test_set_dir , 10 + C20_TRAIN_NUM , test_out_idx , C20_TEST_NUM)
    training_out_idx += C20_TRAIN_NUM
    test_out_idx += C20_TEST_NUM
    
    dir_path = os.path.join(base_dir , "C000022")
    move(dir_path , target_training_set_dir , 10 , training_out_idx , C22_TRAIN_NUM)
    move(dir_path , target_test_set_dir , 10 + C22_TRAIN_NUM , test_out_idx , C22_TEST_NUM)
    training_out_idx += C22_TRAIN_NUM
    test_out_idx += C22_TEST_NUM
   
    print >> sys.stderr , training_out_idx 
    print >> sys.stderr , test_out_idx

    set_type(target_training_set_dir , 1 , C14_TRAIN_NUM , POSITIVE_TYPE_NAME)
    set_type(target_training_set_dir , C14_TRAIN_NUM + 1 , training_out_idx -1 - C14_TRAIN_NUM, NEGATIVE_TYPE_NAME)

    set_type(target_test_set_dir , 1 , C14_TEST_NUM , POSITIVE_TYPE_NAME)
    set_type(target_test_set_dir , 1 + C14_TEST_NUM , test_out_idx - 1 - C14_TEST_NUM , NEGATIVE_TYPE_NAME)
if __name__ == "__main__" :
    if len(sys.argv) != 3 :
        logging.error("usage: %s [%s] [%s]" %(sys.argv[0] , "source_dir" , "target_dir"))
        exit(0)
    ready_data_set(sys.argv[1] , sys.argv[2])

