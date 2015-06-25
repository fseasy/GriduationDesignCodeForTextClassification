#/usr/bin/env python
#coding=utf-8

### Copy From NBSVM 

import logging

#from fileprocessing import POSITIVE_LABEL , NEGATIVE_LABEL
from load_data import POSITIVE_LABEL , NEGATIVE_LABEL
def calc_prf(Y , P_Y) :
    assert len(Y) == len(P_Y)
    TP = 0
    TN = 0 
    FP = 0
    FN = 0
    for i in range(len(Y)) :
        y = float(Y[i])
        py = float(P_Y[i])
        if y == py :
            if y == POSITIVE_LABEL :
                TP += 1
            elif y == NEGATIVE_LABEL :
                TN += 1
            else :
                logging.warn("NOT valid label : '%d' , '%d'" %(py , y))
        else :
            if py == POSITIVE_LABEL :
                FP += 1
            elif py == NEGATIVE_LABEL :
                FN += 1
            else :
                logging.warn("NOT valid label : '%d' , '%d'" %(py , y))
    p_p = 0
    r_p = 0
    f_p = 0
    if TP + FP == 0 :
        # NO Positive predicted , p get 0 or 1 depends on whether dataset has positive instance
        if TP + FN == 0 :
            p_p = 1
        else :
            p_p = 0
    else :
        p_p = TP / float(TP + FP)
    if TP + FN == 0 :
        if TP + FP == 0 :
            r_p = 1
        else :
            r_p = 0
    else :
        r_p = TP / float(TP + FN)
    if p_p + r_p == 0 :
        f_p = 0
    else :
        f_p = 2*p_p*r_p / (p_p + r_p)
    
    p_n = 0
    r_n = 0
    f_n = 0
    if TN + FN == 0 :
        if TN + FP == 0 :
            p_n = 1 
        else :
            p_n = 0
    else :
        p_n = TN / float( TN + FN)
    if TN + FP == 0 :
        if TN + FN == 0 :
            r_n = 1
        else :
            r_n = 0
    else :
        r_n = TN / float( TN + FP)
    if p_n + r_n == 0 :
        f_n = 0 
    else :
        f_n = 2 * p_n * r_n / ( p_n + r_n)

    return (p_p , r_p , f_p) , (p_n , r_n , f_n)

def calc_acc(Y,PY) :
    assert len(Y) == len(PY)
    assert len(Y) > 0
    right_cnt = 0
    total_cnt = len(Y)
    for idx in range(total_cnt) :
        if Y[idx] == PY[idx] :
            right_cnt += 1

    return right_cnt / float(total_cnt)

if __name__ == "__main__" :
    calc_prf([1,-1,1,-1],[1,1,1,-1])
