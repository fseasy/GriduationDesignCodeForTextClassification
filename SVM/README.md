### using SVM classifier (liblinear)

#### tfidf feature

svmtfidf_test.py

输入训练集、测试集、liblinear参数完成SVM训练及测试结果 . 

通过python svmtfidf_test.py --help 查看

'''
sample cmd : python svmtfidf_test.py -ptrain data/imdb/train/postrain.imdb -ptest data/imdb/test/postest.imdb -ntrain data/imdb/train/negtrain.imdb  -ntest data/imdb/test/negtest.imdb -g 1 -c 1 -b -1 -w_p 1 -w_n 1
'''

#### bool feature

svmbool_test.py

有两种模式，一种是交叉验证模式，一种是训练集测试集模式。

