### 本科毕设 代码

####论文题目

 面向不均匀类别的文本分类系统设计与实现

####论文摘要 

本文以设计并实现面向不均匀类别的文本分类系统为目的，主要调研了传统文本分类模型和基于深度学习的分类方法，同时关注数据不均衡问题。在足够的理论基础上，结合开源资源完成实验设计与代码编写，并从实验结果中总结出面向不均匀类别的文本分类方法。本论文主要包含现有方法调研和实验结果，并最终给出了不均匀数据下的文本分类方法设计，公开了实验代码。

####代码

主要包含特征抽取的脚本。

BOW是实现基本的BOW模型。

SVM是完成bool , tfidf , NB特征抽取及NBSVM论文实现

BOOSTEDTREE完成bool,tfidf,NB特征抽取及存储。

NGRAM_EMBEDDING完成Bag-Of-SemanticConcept方法实现。使用word embedding做特征降维。

SMOTE是人工构建新特征方法的实现。

CNN是依据CNN_sentence项目和deepling tutorial写的代码。使用了theano.


