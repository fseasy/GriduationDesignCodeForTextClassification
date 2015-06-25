#!/usr/bin/env python
#coding=utf8

import subprocess
import os
import re
import logging 

logging.basicConfig(
        level=logging.DEBUG ,
        format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :: "%(message)s"' ,
        datefmt='%H:%M:%S'
        )

def seg_word() :
    seg_proc = subprocess.Popen("/users1/exe/projects/ltp/bin/examples/cws_cmdline /data/ltp/ltp-models/3.2.0-server/ltp_data/cws.model" , stdin=subprocess.PIPE , stdout=subprocess.PIPE , shell=True)

    #seg_proc.stdin.write("什么鬼！\n") ;
    #seg_proc.stdin.flush()
    #print seg_proc.stdout.readline() 
    #print "over"
    dir_path="sample_data/C08"
    file_list = os.listdir(dir_path)
    for file_name in file_list :
        logging.debug("file_name : %s" %(file_name))
        path = os.path.join(dir_path , file_name)
        read_file = open(path)
        content = read_file.read()
        content = content.decode("gb18030")
        content = content.replace("&nbsp","") ;
        #content = re.sub(ur'\s' , '\n' , content)
        #print repr(content)
        content = re.sub(ur'(?<=\n)[ \u3000]+' , '', content) # 全角空格或空格
        content = re.sub(ur'^[ \u3000]+' , '\n', content)
        content = content.encode("utf-8")
        seg_proc.stdin.write(content + "\n")
        seg_proc.stdin.flush()
        content = ""
        while True :
            content += seg_proc.stdout.readline()
            if content == "" :
                break
        out_file = open(file_name , 'w')
        out_file.write(content) ;
        out_file.close()
        read_file.close()

if __name__ == "__main__" :
    seg_word()
