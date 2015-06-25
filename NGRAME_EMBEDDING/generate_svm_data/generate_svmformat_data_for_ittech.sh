#/bin/sh
echo "ensure has set the IT_Tech dict data and other path ?(y/n) " >/dev/stderr
read input
if [ $input != 'y' ]; then
    echo "exit" >/dev/stderr
    exit 1
fi
#######IT_Tech banlance uni
#data_type="itech_uni"
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/ittech_uni.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/ittech_uni/ittech_uni.assignments_k200"
#
#
#IT_Tech_POS_TRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/pos.train"
#IT_Tech_NEG_TRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/neg.train"
#IT_Tech_POS_TEST="/users1/wxu/code/text_category/SVM/data/IT_Tech/test/pos.test"
#IT_Tech_NEG_TEST="/users1/wxu/code/text_category/SVM/data/IT_Tech/test/neg.test"

####### IT_Tech 10vs1 uni
#data_type="ittech10vs1_uni"
#DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/ittech10vs1_uni.dict.data"
#ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/ittech10vs1_uni/ittech10vs1_uni.assignments_k200"
#
#IT_Tech_POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.train"
#IT_Tech_NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.train.1650"
#IT_Tech_POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.test"
#IT_Tech_NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.test.550"

####### IT_Tech 10vs1 uni_bi

data_type="ittech10vs1_unibi"
DICT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/doc_dict_data/ittech10vs1_uni_bi.dict.data"
ASSIGNMENT_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/ready_word_vector/cluster_result/ittech10vs1_uni_bi/ittech10vs1_uni_bi.assignments_k200"

IT_Tech_POS_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.train"
IT_Tech_NEG_TRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.train.1650"
IT_Tech_POS_TEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.test"
IT_Tech_NEG_TEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.test.550"


K="200"
out_dir="data/$data_type"
mkdir -p $out_dir
MODEL_OUT_F="$out_dir/${data_type}.model"
IT_Tech_POS_TRAIN_OUT="$out_dir/pos.train.svm"
IT_Tech_NEG_TRAIN_OUT="$out_dir/neg.train.svm"
IT_Tech_POS_TEST_OUT="$out_dir/pos.test.svm"
IT_Tech_NEG_TEST_OUT="$out_dir/neg.test.svm"
IT_Tech_TRAIN_OUT="$out_dir/${data_type}.train.svm"
IT_Tech_TEST_OUT="$out_dir/${data_type}.test.svm"
python ready_dimension_reduction_model.py --dictpath $DICT_DATA --cluster_assment $ASSIGNMENT_DATA --cluster_num $K --out $MODEL_OUT_F
python generate_svm_data.py --rawdata $IT_Tech_POS_TRAIN --label 1 --out $IT_Tech_POS_TRAIN_OUT --drmodel $MODEL_OUT_F
python generate_svm_data.py --rawdata $IT_Tech_NEG_TRAIN --label 0 --out $IT_Tech_NEG_TRAIN_OUT --drmodel $MODEL_OUT_F

python generate_svm_data.py --rawdata $IT_Tech_POS_TEST --label 1 --out $IT_Tech_POS_TEST_OUT --drmodel $MODEL_OUT_F
python generate_svm_data.py --rawdata $IT_Tech_NEG_TEST --label 0 --out $IT_Tech_NEG_TEST_OUT --drmodel $MODEL_OUT_F

cat $IT_Tech_POS_TRAIN_OUT $IT_Tech_NEG_TRAIN_OUT > $IT_Tech_TRAIN_OUT
cat $IT_Tech_POS_TEST_OUT $IT_Tech_NEG_TEST_OUT > $IT_Tech_TEST_OUT

rm $IT_Tech_POS_TRAIN_OUT $IT_Tech_NEG_TRAIN_OUT $IT_Tech_POS_TEST_OUT $IT_Tech_NEG_TEST_OUT
