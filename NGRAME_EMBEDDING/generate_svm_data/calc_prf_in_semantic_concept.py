#/usr/bin/env python
#coding=utf-8

import argparse

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

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

def main(tf , pf) :
    tf_list = []
    pf_list = []
    counter = 0
    while True :
        tf_line = tf.readline().strip()
        pf_line = pf.readline().strip()
        
        if tf_line == "" :
            try:
                assert pf_line == ""
            except AssertionError , e :
                print "tf , pf has different line num"
                traceback.print_exc()
                exit(1)
            break
        counter += 1
        tf_list.append(int(tf_line))
        pf_list.append(int(pf_line))
    print "label num : %d" %(counter)
    p_prf , n_prf = calc_prf(tf_list,pf_list)
    print "positive class : prf=%s\nnegative class : prf=%s" %(p_prf , n_prf)


if __name__ == "__main__" :
    argp =argparse.ArgumentParser(description="caclc prf in semantic concept")
    argp.add_argument("-tf" , help="path to true result label file" , type=argparse.FileType('r') , required=True)
    argp.add_argument('-pf' , help="path to predict result label file " , type=argparse.FileType('r') , required=True)

    args = argp.parse_args()
    main(args.tf , args.pf)
    args.tf.close()
    args.pf.close()
