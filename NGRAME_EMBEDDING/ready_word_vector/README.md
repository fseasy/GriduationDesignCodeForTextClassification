###READY A TRANS DICT ( Ngram -> class(theme id) ) 

#### train_word2vec.sh

训练生成word2vec向量

#### run_build_dict_and_trans.sh
     
\-build_dict_and_trans2vector.py 

从docs中抽取ngram词典，利用上述的word2vec向量将词典全部向量化

#### run_cluster_dictvector_using_sofiaml.sh

使用sofia-ml将上述向量化的ngram词进行聚类。

sofia-ml的输入时libsvm格式的数据，需要先给一个label，再给出idx:val，这比较奇怪，应该这个工具不是单纯用于聚类的

转换后使用工具先获得K个聚类中心保留下来，再利用刚刚的聚类中心确定每个原始向量所属的类别，输出的每一行，第一列是中心序号，第二列没用。每行与词典顺序对应，到此就完成了聚类。每个词对应一个类别(theme id)，使用该类别作为特征项。

#### run_all.sh

将上述流串起来
