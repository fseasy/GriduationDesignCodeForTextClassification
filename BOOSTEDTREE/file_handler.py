#/usr/bin/env python
#coding=utf8

from collections import Counter

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

def stat_oneclass_docs(docs_f , docs_label , gram_num) :
    '''
    return > this class' docs words and TF , [ [(wrod , TF) ,] ] 
             class labels , [ label , ..]
             class' all words and 'DF' , Counter
    '''
    class_counter = Counter()
    class_docs = []
    class_labels = []
    for doc in docs_f :
        all_words_list = tokenize(doc , gram_num)
        doc_counter = Counter(all_words_list)
        class_docs.append(doc_counter.items())
        class_counter.update(doc_counter.keys())
        class_labels.append(docs_label)
    return class_docs , class_labels , class_counter

def build_docs_dict(class_counter_list) :
    dict_counter = Counter()
    for item in class_counter_list :
        dict_counter.update(item.keys())
    return dict_counter.keys()

def save_in_libsvm_dense_format(Y,X,dimension,file_path) :
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

def save_in_libsvm_sparse_format(Y,X,ofi) :
    '''
    X -> [ {1:0.1,5:9,...}     ]
    '''
    
    is_f = isinstance(ofi , file)
    if not is_f :
        ofi = open(ofi , 'w')
    for y,x in zip(Y,X) :
        line = [str(y) ,]
        x = sorted(x.items() , key=lambda t : t[0])
        x_str = ["%d:%f" %(idx,val) for idx , val in x]
        line.extend(x_str)
        line_str = " ".join(line)
        ofi.write(line_str + '\n')
    if not is_f :
        ofi.close()

def load_for_libsvm_sparse_format(ifi) :
    '''
    :param ifi : input file for stored sparse format data . 
                 like 
                    label idx:val idx:val ...
                    label idx:val idx:val ...
                    ...
    :return X : [ {idx:val,idx:val,idx:val} , {idx:val,idx:val,...} ... ]
    :return Y : [ label                     , label                 ... ]
    '''
    is_f = isinstance(ifi , file)
    if not is_f :
        ifi = open(ifi)
    X = []
    Y = []
    for line in ifi :
        x_instance = {}
        parts = line.strip().split(' ') # specific the space as the delimiter
        Y.append(int(parts[0]))
        for i in range(1,len(parts)) :
            idx , val = parts[i].split(':')
            idx = int(idx)
            val = float(val)
            if val == 0.0 : continue
            x_instance[idx] = val
        X.append(x_instance)
    return X , Y

