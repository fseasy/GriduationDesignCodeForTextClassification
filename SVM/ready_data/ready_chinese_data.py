#!/usr/bin/env python
#coding=utf-8


import logging
import os
import sys
import argparse

def main(indir , outdir) :
    is not os.path.isdir(indir) :
        logging.error("%s does not a valid dir")
        return 0
    f_list = 

if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(desription="ready chinese data")
    argparser.add_argument("--indir" , help="path to input dir" , type=str , required=True)
    argparser.add_argument("--outdir" , help="path to output dir" , type=str , required=True)
    args = argparser.parse_args()

    if not os.path.exists(args.indir) :
        logging.error(e)
        exit(0)
    if not os.path.exits(args.outdir) :
        print >> sys.stderr , "output dir '%s' does not exists .\n create it ?[y/n] " 
        r = raw_input()
        if r == "y" :
            try :
                os.makedirs(outdir)
            except Exception , e :
                logging.error(e)
                exit(0)
        else :
            exit(0)
    main(**vars(args))
