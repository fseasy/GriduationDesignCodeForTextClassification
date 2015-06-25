#/bin/sh

### Data Config

################ IMDB banlance uni
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdb_uni.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdb_uni/imdb.assignments_k200"
#
#DATA_NAME="imdb_uni"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/data/imdb/train/postrain.imdb"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/data/imdb/train/negtrain.imdb"
#POS_TEST="/users1/wxu/code/text_category/SVM/data/imdb/test/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/data/imdb/test/negtest.imdb"

################ IMDB 10vs1 uni
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdb10vs1_uni.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdb10vs1_uni.dict/imdb10vs1_uni.assignments_k200"
#DATA_NAME="imdb10vs1_uni"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/postrain.imdb"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/negtain.imdb.1250"
#POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/negtest.imdb.1250"

################ IMDB 10vs1 uni_bi

DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdb10vs1_uni_bi.dict.data"
ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdb10vs1_uni_bi/imdb10vs1_uni_bi.assignments_k200"
DATA_NAME="imdb10vs1_uni_bi"

POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/postrain.imdb"
NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/negtain.imdb.1250"
POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/postest.imdb"
NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/negtest.imdb.1250"

################ IMDB oversampling 10vs1->1vs1 uni

#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdbos1vs1_uni.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdbos1vs1_uni/imdbos1vs1_uni.assignments_k200"
#DATA_NAME="imdbos1v1_uni"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postrain.imdb"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtrain.imdb.12500.random"
#POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtest.imdb.1250"

############### IMDB oversampling 10vs1->1vs1 uni_bi

#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdbos1vs1_uni_bi.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdbos1vs1_uni_bi/imdbos1vs1_uni_bi.assignments_k200"
#DATA_NAME="imdbos1v1_uni_bi"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postrain.imdb"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtrain.imdb.12500.random"
#POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtest.imdb.1250"

############### IMDB undersampling 10vs1->1vs1 uni
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdbus1vs1_uni.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdbus1vs1_uni/imdbus1vs1_uni.assignments_k200"
#DATA_NAME="imdbus1vs1_uni"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postrain.imdb.1250.random"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtrain.imdb.1250"
#POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtest.imdb.1250"

############### IMDB undersampling 10vs1->1vs1 uni_bi
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/imdbus1vs1_uni_bi.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/imdbus1vs1_uni_bi/imdbus1vs1_uni_bi.assignments_k200"
#DATA_NAME="imdbus1vs1_uni_bi"
#
#POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postrain.imdb.1250.random"
#NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtrain.imdb.1250"
#POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postest.imdb"
#NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtest.imdb.1250"


K="200"
PRE_PATH="data/$DATA_NAME"
mkdir -p $PRE_PATH

POS_TRAIN_OUT="$PRE_PATH/pos.train.svm"
NEG_TRAIN_OUT="$PRE_PATH/neg.train.svm"
POS_TEST_OUT="$PRE_PATH/pos.test.svm"
NEG_TEST_OUT="$PRE_PATH/neg.test.svm"

MODEL_OUT_F="$PRE_PATH/${DATA_NAME}.dr.model"
SVM_TRAIN_OUT="$PRE_PATH/${DATA_NAME}.train.svm"
SVM_TEST_OUT="$PRE_PATH/${DATA_NAME}.test.svm"

python ready_dimension_reduction_model.py --dictpath $DICT_DATA --cluster_assment $ASSIGNMENT_DATA --cluster_num $K --out $MODEL_OUT_F
python generate_svm_data.py --rawdata $POS_TRAIN --label 1 --out $POS_TRAIN_OUT --drmodel $MODEL_OUT_F
python generate_svm_data.py --rawdata $NEG_TRAIN --label 0 --out $NEG_TRAIN_OUT --drmodel $MODEL_OUT_F

python generate_svm_data.py --rawdata $POS_TEST --label 1 --out $POS_TEST_OUT --drmodel $MODEL_OUT_F
python generate_svm_data.py --rawdata $NEG_TEST --label 0 --out $NEG_TEST_OUT --drmodel $MODEL_OUT_F

cat $POS_TRAIN_OUT $NEG_TRAIN_OUT > $SVM_TRAIN_OUT
cat $POS_TEST_OUT $NEG_TEST_OUT > $SVM_TEST_OUT

rm $POS_TRAIN_OUT $NEG_TRAIN_OUT $POS_TEST_OUT $NEG_TEST_OUT

