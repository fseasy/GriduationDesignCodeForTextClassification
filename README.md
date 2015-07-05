### 本科毕设 代码

####论文题目

 面向不均匀类别的文本分类系统设计与实现

####论文摘要 

本文以设计并实现面向不均匀类别的文本分类系统为目的，主要调研了传统文
本分类模型和基于深度学习的分类方法，包括Bag-Of-Words 模型，Ngram（unigram
和unigram 与bigram 组合）表示效果，布尔、归一化TFIDF、NB 特征表示，SVM
和Boosting 方法，以及卷积神经网络模型。同时关注数据不均衡下的处理方法，
主要涉及过采样、下采样，调整分类器参数及SMOTE 方法。在足够的理论基础上，
结合开源资源完成实验设计与代码编写。在英文IMDB 数据集和搜狗IT 和科技类
别数据集上完成均匀和不均匀数据的分类效果测试，在IMDB 数据集上比较了过采
样、下采样，SVM 调参与SMOTE 方法的实际效果。最后通过对实验结果的分析，从
中总结出面向不均匀类别的文本分类系统的设计方案。设计1 使用BOOSTING 方法，
使用文本的unigram 布尔特征，对不均衡数据集做过采样处理，构建出稳定适应性
强的分类系统。设计2 使用unigram 与bigram 组合及归一化TFIDF 特征，以SVM
作为分类方法，通过调整SVM 参数完成特定数据集上的分类系统构建。实验的实
现代码已公布在互联网上，主要包含各方法的特征抽取代码和工作流脚本。

####代码

主要包含特征抽取的脚本。

BOW是实现基本的BOW模型。

SVM是完成bool , tfidf , NB特征抽取及NBSVM论文实现

BOOSTEDTREE完成bool,tfidf,NB特征抽取及存储。

NGRAM_EMBEDDING完成Bag-Of-SemanticConcept方法实现。使用word embedding做特征降维。

SMOTE是人工构建新特征方法的实现。

CNN是依据CNN_sentence项目和deepling tutorial写的代码。使用了theano.


