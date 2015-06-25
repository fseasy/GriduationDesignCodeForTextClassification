#!/usr/bin/env python
#coding=utf-8
import os
import sys

import traceback
import logging

stop_words = {}


def load_stop_words(stop_words_dir_name) :
    if not os.path.isdir(stop_words_dir_name) :
        return
    file_list = os.listdir(stop_words_dir_name)
    for file_name in file_list :
        try :
            file_path = os.path.join(stop_words_dir_name , file_name)
            read_file = open(file_path)
            for line in read_file :
                line = line.strip()
                try :
                    line = line.decode("utf-8")
                    stop_words[line] = 0
                except Exception , e :
                    loggging.warning("exception occured at loading stop words. May be decoding error . \n %s " %(e))
        except Exception , e :
            logging.warning("exception occured at loading stop words . May be open file failed . \n %s " %(e))
            logging.warning(traceback.print_exc())
        finally :
            read_file.close()

def read_words_recursive(dir_or_file_name , words_dict) :
    if not os.path.exists(dir_or_file_name) :
        return
    elif os.path.isfile(dir_or_file_name) :
        try :
            read_file = open(dir_or_file_name)
            for line in read_file.readlines() :
                line = line.strip()
                try :
                    line = line.decode("utf-8")
                except Exception , e :
                    logging.warning("read words decode error : \n"  %(e))
                    continue
                line_list = line.split()
                for word in line_list :
                    if word not in stop_words :
                        if word not in words_dict :
                            words_dict[word] = 1
                        else :
                            words_dict[word] += 1
                    else :
                        stop_words[word] += 1
        except Exception , e :
            logging.warning("faild to read file : ' %s '\n %s " %(dir_or_file_name , e))
        finally :
            read_file.close()
    else :
        file_list = os.listdir(dir_or_file_name ) 
        for file_name in file_list :
            cur_path = os.path.join(dir_or_file_name , file_name)
            read_words_recursive(cur_path , word_dict)

if __name__ == "__main__" :
    if len(sys.argv) != 3 :
        logging.error("usage: %s [%s] [%s] " %(sys.argv[0] , "stop_words_dir" , "readed_dir"))
        exit(0)
    load_stop_words(sys.argv[1])
    word_dict = {}
    read_words_recursive(sys.argv[2] , word_dict)

    #write the result to the file
    out_stop_words_f = open("stop_words_work_info" , "w")
    sorted_list = sorted(stop_words.items() , key=lambda x : x[1] , reverse=True)
    for item in sorted_list :
        key_out = item[0].encode("utf-8")
        out_stop_words_f.write("%s\t%d\n"  %(key_out , item[1]))
    out_stop_words_f.close()
    
    sorted_list = sorted(word_dict.items() , key=lambda x : x[1] , reverse=True)
    out_read_words_f = open("read_words_info" , "w")
    for item in sorted_list :
        key_out = item[0].encode("utf-8")
        out_read_words_f.write("%s\t%d\n"  %(key_out , item[1]))
    out_read_words_f.close()
