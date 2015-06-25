#/usr/bin/env python
#coding=utf-8

import sys
import argparse
try :
    import cPickle as pickle
except :
    import pickle

def main(f) :
    rst = pickle.load(f)
    for item in rst :
        o_l = []
        o_l.append(item['c'])
        o_l.append(item['w_positive'])
        o_l.append(item['w_negative'])
        o_l.append(item['bias'])
        if 'beta' in item : o_l.append(item['beta'])
        o_l.append(item['negative_prf'][2])
        o_l = [ '%f' %(x) for x in o_l ]
        print "\t".join(o_l)
    sorted_rst = sorted(rst , key=lambda x : x['negative_prf'][2] , reverse=True)
    limit= min(10 , len(sorted_rst))

    print >> sys.stderr , "按照F值从高到低排列前%d位结果为:" %(limit)
    for i in range(limit) :
        item = sorted_rst[i]
        print >> sys.stderr , "c = %f , w_p = %.2f , w_n = %.2f , bias = %.2f " %(item['c'] , item['w_positive'] , item['w_negative'] , item['bias']) ,
        if 'beta' in item :
            print >> sys.stderr , "beta = %.2f " %(item['beta']),
        print >> sys.stderr , "f = %.2f %%" %(item['negative_prf'][2])




if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="From grid pickle result print the parames and result values")
    argp.add_argument("-f" , "--rst_f" , help="path to grid pickle file" , required=True , type=argparse.FileType('r'))

    args = argp.parse_args()
    main(args.rst_f)
    
    args.rst_f.close()
