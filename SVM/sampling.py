#/usr/bin/env python
#coding=utf-8
import argparse
import sys
import random

def main(src_f , target_f , num) :
    lines = src_f.readlines()
    total_num = len(lines)
    for i in range(0 , num) :
        sample = random.choice(lines)
        target_f.write(sample)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="random samping dataset")
    argp.add_argument("--src" , required=True , help="source dataset file" , type=argparse.FileType("r"))
    argp.add_argument("--target" , required=True , help="path to saved the sampling result" , type=argparse.FileType("w"))
    argp.add_argument("--num" , required=True , help="num of random sampling time" , type=int)
    args = argp.parse_args()
    main(args.src , args.target , args.num) 
    args.src.close()
    args.target.close()

