#/usr/bin/env python
#coding=utf-8

import sys
print >> sys.stderr , 'read from stdin'
lengths = []
for line in sys.stdin :
    line = line.strip()
    lengths.append(len(line))
print 'Max length : %d \nMin length : %d \nAverage Length : %d' %(max(lengths) , min(lengths) , sum(lengths)/len(lengths))
