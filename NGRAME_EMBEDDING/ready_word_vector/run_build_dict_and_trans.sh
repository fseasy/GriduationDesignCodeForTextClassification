#/bin/sh 
TYPEs=("IT_Tech" "IMDB")
grams=("uni" "uni_bi")

if [ $# -ne 2 ]; then 
    echo "$0 [TYPE:${TYPEs[0]}/${TYPEs[1]}] [gram:${grams[0]}/${grams[1]}]" >/dev/stderr
    exit 1
fi

PREPATH="doc_dict_data"
mkdir -p $PREPATH

TYPE="$1"
gram="$2"
valid="False"
if echo "${TYPEs[@]}" | grep -w "$TYPE" &>/dev/null 
   echo "${grams[*]}" | grep -w "$gram" &>/dev/null
then
    valid="True"
else 
    echo "parameter invalid : ' ${TYPE} ' not in ""'""${TYPEs[@]}""'" "or '${gram}' not in "\'"${grams[*]}"\'
    exit 1
fi

###################Test
#TEST_POSTRAIN="/users1/wxu/code/text_category/NGRAME_EMBEDDING/test_data/IT_tech/train/pos.train.825"
#TEST_NEGTRAIN="/users1/wxu/code/text_category/NGRAME_EMBEDDING/test_data/IT_tech/train/neg.train.825"
#TEST_WORDVECTOR="/users1/wxu/data/word2vector/cn_for_test/wordvector.test.txt"
#TEST_DICT_DATA="dict.data.test"
#TEST_KMEANS_OUT="classinfo.data" 
#TEST_KMEANS_DATA="/users1/wxu/code/text_category/NGRAME_EMBEDDING/test_data/kmeans_data/testSet.txt" # for test kmeans

if [ "$TYPE" == "IT_Tech" ]
then
### IT_Tech balance
#POSTRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/pos.train"
#NEGTRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/neg.train"
#WORDVECTOR="/users1/wxu/data/word2vector/cn/sogou.word2vec.txt"
#DICT_DATA="${PREPATH}/ittech_${gram}.dict.data"

### IT_Tech 10vs1
POSTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.train"
NEGTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.train.1650"
WORDVECTOR="/users1/wxu/data/word2vector/cn/sogou.word2vec.txt"
DICT_DATA="${PREPATH}/ittech10vs1_${gram}.dict.data"


else

#################### IMDB balance ##
#POSTRAIN="/users1/wxu/code/text_category/SVM/data/imdb/train/postrain.imdb"
#NEGTRAIN="/users1/wxu/code/text_category/SVM/data/imdb/train/negtrain.imdb"
#WORDVECTOR="/users1/wxu/data/word2vector/en/1billion.wordvectors"
#DICT_DATA="${PREPATH}/imdb_${gram}.dict.data"
###
### IMDB 10vs1
#POSTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/postrain.imdb"
#NEGTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/negtain.imdb.1250"
#WORDVECTOR="/users1/wxu/data/word2vector/en/1billion.wordvectors"
#DICT_DATA="${PREPATH}/imdb10vs1_${gram}.dict.data"

### IMDB oversampling 10vs1->1vs1
#POSTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postrain.imdb"
#NEGTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtrain.imdb.12500.random"
#WORDVECTOR="/users1/wxu/data/word2vector/en/1billion.wordvectors"
#DICT_DATA="${PREPATH}/imdbos1vs1_${gram}.dict.data"

### IMDB undersampling 10vs1->1vs1
POSTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postrain.imdb.1250.random"
NEGTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtrain.imdb.1250"
WORDVECTOR="/users1/wxu/data/word2vector/en/1billion.wordvectors"
DICT_DATA="${PREPATH}/imdbus1vs1_${gram}.dict.data"

fi

if [ "$gram" == "${grams[0]}" ]
then
    ngram=1
else
    ngram=2
fi

echo "build dict for $TYPE in $ngram gram"

python build_dict_and_trans2vector.py --postrain $POSTRAIN --negtrain $NEGTRAIN --ngram $ngram --wordvector $WORDVECTOR --out $DICT_DATA

echo "SAVE DICT_DATA to $DICT_DATA"
