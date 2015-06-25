#!/usr/bin/env python
#coding=utf-8

from collections import Counter

NGRAM_JOIN_CHAR="_"

def unordered_ngram_tokenizer(line , ngram_n) :
    """
    input > line : a doc line
            ngram_n : 1,2,3,and so on
    return > Counter
    """
    
    c = Counter()
    parts = line.strip().split()
    #unigram
    c.update(parts)
    parts_num = len(parts)
    for n in range(2,ngram_n+1) :
        for i in range(0,parts_num - n + 1) :
            ngram = parts[i:i+n]
            ngram_unordered = sorted(ngram)
            ngram_str = NGRAM_JOIN_CHAR.join(ngram_unordered)
            c[ngram_str] += 1
    return c


if __name__ == "__main__" :
    l = "我 是 中国 人 。"
    c = unordered_ngram_tokenizer(l , 2)
    for key , val in c.items() :
        print "%s,%d" %(key , val)

