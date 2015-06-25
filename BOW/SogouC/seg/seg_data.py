#!/bin/usr/env python

import sys
import os
import re
import traceback
import subprocess
import logging
logging.basicConfig(
            level=logging.DEBUG 
        )

def word_seg_recursive(cur_name , target_name) :
    if not os.path.exists(cur_name) :
        return
    if os.path.isfile(cur_name) :
        target_dir = os.path.split(target_name)[0]
        if not os.path.exists(target_dir) :
            os.makedirs(target_dir)
        read_file = open(cur_name)
        write_file = open(target_name , 'w')
        seg_proc = subprocess.Popen("/users1/exe/projects/ltp/bin/examples/cws_cmdline /data/ltp/ltp-models/3.2.0-server/ltp_data/cws.model" , stdin=subprocess.PIPE , stdout=write_file , shell=True)
        content = ""
        try :
            content = read_file.read()
            content = content.decode("gb18030")
            content = content.replace("&nbsp","")
            content = re.sub(ur'(?<=\n)[ \u3000]+' , '', content)
            content = re.sub(ur'^[ \u3000]+' , '\n', content)
            content = content.encode("utf-8")
        except Exception , e :
            logging.debug(e)
            logging.debug(traceback.print_exc())
        seg_proc.stdin.write(content)
        seg_proc.stdin.close()
        seg_proc.wait()
        write_file.close()
        read_file.close()
    else :
        lists = os.listdir(cur_name)
        for file_name in lists :
            cur_path = os.path.join(cur_name , file_name)
            target_path = os.path.join(target_name , file_name)
            word_seg_recursive(cur_path , target_path)



def seg_data_main():
    if len(sys.argv) != 3 :
        print sys.argv
        print >> sys.stderr , "usage : %s [%s] [%s] " %(sys.argv[0] , "source_path" , "target_path")
        return 0
    else :
        # check path
        if not os.path.exists(sys.argv[1]) :
            print >> sys.stderr , "source path : '%s' does not exists" %( sys.argv[1])
            return 0
        if not os.path.exists(sys.argv[2]) :
            print >> sys.stderr , "target path : '%s' does not exists" %(sys.argv[2])
            tips = "create '%s' ? [y/n]" %(sys.argv[2])
            res = raw_input(tips)
            if res == 'y' :
                try :
                    os.makedirs(sys.argv[2])
                except Exception , e :
                    print >> sys.stderr , e 
                    print >> sys.stderr , traceback.print_exc()
                    return 0
            else :
                return 0 
        word_seg_recursive(sys.argv[1] , sys.argv[2])

if __name__ == "__main__" :
    seg_data_main() 
