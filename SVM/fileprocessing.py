#!/usr/bin/env python
#coding=utf-8

#global parameter
POSITIVE_LABEL = 1
NEGATIVE_LABEL = -1

def tokenize(sentence , ngram) :
    words = sentence.split()
    cnt = len(words)
    tokens = []
    # get gram from 1 to ngram
    for gram in range(1,ngram+1) :
    #gram = ngram
        for i in range(0,cnt - gram + 1) :
            token_l = words[i:i+gram]
            tokens.append('_'.join(token_l))
    return tokens

def tokenize_including_low(sentence , ngram) :
    '''
    get gram from unigram to ngam , not only the ngram
    '''
    words = sentence.split()
    cnt = len(words)
    tokens = []
    for gram in range(1,ngram+1) :
        for i in range(1,cnt - gram + 1) :
            token_l = words[i:i+gram]
            tokens.append('_'.join(token_l))
    return tokens

def vectorize_docs(f_obj , dic , r , ngram) :
    f_vector = [] 
    for line in f_obj.xreadlines() :
        tokens = tokenize(line , ngram)
        # for this , just need the occurence of every token
        tokens = list(set(tokens))
        index = []
        for token in tokens :
            if token in dic :
                index.append(dic[token])
        index.sort() # let the index arranged from small to big
        line_vector = []
        for i in index :
            if r[i-1] != 0 : # if r[i] = 0 , no need to output . because we build the sparse data format
                line_vector.append((i,r[i-1])) # f = r * f_ori 
        f_vector.append(line_vector)
    return f_vector


def ready_SVM_data(labels , vecs ) :
    '''
    input >  labels : list for labels of every class
             vecs   : vectors of every class
    return > Y,X : labels and vectors for every case !

    pack it !
    '''
    Y = []
    X = []
    for label , f_vec in zip(labels , vecs) :
        for x in f_vec :
            Y.append(label)
            X.append({ idx:val for idx,val in x})
    return Y , X

def save_SVM_data(Y,X,dimension,file_path) :
    f = open(file_path,'w')
    sample_line = ['%d:%f' %(i,0.0) for i in range(1,dimension+1)]
    for y,x in zip(Y,X) :
        line = [str(y)] + sample_line
        for idx in x :
            #line.append("%d:%f" %(idx,x[idx]))
            line[idx] = "%d:%f" %(idx,x[idx])
        line = ' '.join(line) + '\n'
        f.write(line)
    f.close()


def output_predict_detail(doc_id_l , ty_l , py_l , p_outf) :
    assert(len(doc_id_l) == len(ty_l) == len(py_l))
    for i in range(len(doc_id_l)) :
        p_outf.write("%10s\t%10d\t%10d\n" %(doc_id_l[i] , ty_l[i] , py_l[i]))


if __name__ == "__main__" :
    dic = {
            "i":1 ,
            "is_a":2
            }
    r = [23,9]
    v = vectorize_docs(open("data/postrain") , dic , r , 2)
    print v
