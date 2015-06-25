#!/usr/bin/env python
#coding=utf-8

import argparse
import logging

POSITIVE_LABEL = "1"
NEGATIVE_LABEL = "-1"


def main(rstf) :
    TP = 0
    FP = 0
    TN = 0 # the true label is positive while prediction is negtative 
    FN = 0
    for line in rstf :
        line_parts = line.strip().split()
        try :
            ty = line_parts[-2]
            py = line_parts[-1]
            if ty == py :
                if ty == POSITIVE_LABEL :
                    TP += 1
                else :
                    TN += 1
            else :
                if py == POSITIVE_LABEL :
                    FP += 1
                else :
                    FN += 1
        except Exception , e :
            logging.error(e)
            return 0
    Pp = float(TP) / (TP + FP)
    Rp = float(TP) / (TP + FN)
    Fp = 2*Pp*Rp / (Pp + Rp)

    Pn = float(TN) / (TN + FN)
    Rn = float(TN) / (TN + FP)
    Fn = 2*Pn*Rn / (Pn + Rn)
    rst_s = """
    TP:%d , FP:%d , TN:%d , FN:%d
    For positive class : P = %.2f , R = %.2f , F1 = %.2f 
    For negative class : P = %.2f , R = %.2f , F1 = %.2f
    """%(
        TP , FP , TN , FN ,
        Pp * 100 , Rp * 100 , Fp * 100 ,
        Pn * 100 , Rn * 100 , Fn * 100
            )
    
    print rst_s

if __name__ == "__main__" :
    argparser = argparse.ArgumentParser(description="analysis the predict result . the reuslt shoul in the format : ... TY[space or TAB]PY")
    argparser.add_argument("--rstf" , help="path to prediction result file" , type=argparse.FileType("r") , required=True)
    args = argparser.parse_args()
    main(args.rstf)
