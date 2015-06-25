#!/usr/bin/env python
#coding=utf-8

import sys
import os
import logging
import argparse

def trans_gb180302utf8(f,o_f) :
    for l in f :
        try :
            l = l.decode("gb18030")
            l = l.encode("utf-8")
            o_f.write(l)
        except (UnicodeEncodeError , UnicodeDecodeError ) , e :
            logging.error(e)

if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(description="trans file from gb18030 to utf-8")
    argparser.add_argument("--stream" , action="store_true")
    argparser.add_argument("--infile" , type=str , default="")
    args = argparser.parse_args()
    if args.stream :
        trans_gb180302utf8(sys.stdin , sys.stdout)
        exit(0)
    if not os.path.exists(args.infile) :
        logging.error("%s doesnot exists" %(args.infile))
        exit(0)
    f_i = open(args.infile)
    f_o = open(args.infile + "_trans" , 'w')
    trans_gb180302utf8(f_i,f_o)
    f_i.close()
    f_o.close()
