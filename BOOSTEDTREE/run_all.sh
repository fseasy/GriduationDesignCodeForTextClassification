#/bin/sh

###IT_Tech banlance
#PTRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/pos.train"
#PTEST="/users1/wxu/code/text_category/SVM/data/IT_Tech/test/pos.test"
#NTRAIN="/users1/wxu/code/text_category/SVM/data/IT_Tech/train/neg.train"
#NTEST="/users1/wxu/code/text_category/SVM/data/IT_Tech/test/neg.test"
#
#OUT_PREFIX="data/ittech"
#
### IMDB 10vs1 
#PTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/postrain.imdb"
#PTEST="/users1/wxu/code/text_category/SVM/tmpdata/postest.imdb"
#NTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/negtain.imdb.1250"
#NTEST="/users1/wxu/code/text_category/SVM/tmpdata/negtest.imdb.1250"
#OUT_PREFIX="data/imdb.10vs1"
#
### IT_Tech 10vs1
#PTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.train"
#PTEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/pos.test"
#NTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.train.1650"
#NTEST="/users1/wxu/code/text_category/SVM/tmpdata/ittech/neg.test.550"
#OUT_PREFIX="data/ittech.10vs1"

### IMDB oversampling 
#PTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postrain.imdb"
#PTEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/postest.imdb"
#NTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtrain.imdb.12500.random"
#NTEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.os.1v1/negtest.imdb.1250"
#OUT_PREFIX="data/imdb.os"

### IMDB undersampling
PTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postrain.imdb.1250.random"
PTEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/postest.imdb"
NTRAIN="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtrain.imdb.1250"
NTEST="/users1/wxu/code/text_category/SVM/tmpdata/imdb.us.1v1/negtest.imdb.1250 "
OUT_PREFIX="data/imdb.us"



echo "genrate bool feature file" >/dev/stderr
BOOL_PREFIX="$OUT_PREFIX/bool"
mkdir -p $BOOL_PREFIX
python trans_in_bool_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 1 -o "${BOOL_PREFIX}/uni.bool"
python trans_in_bool_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 2 -o "${BOOL_PREFIX}/unibi.bool"

echo "genrate tfidf feature file" >/dev/stderr
TFIDF_PREFIX="$OUT_PREFIX/tfidf"
mkdir -p $TFIDF_PREFIX
python trans_in_tfidf_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 1 -o "${TFIDF_PREFIX}/uni.tfidf"
python trans_in_tfidf_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 2 -o "${TFIDF_PREFIX}/unibi.tfidf"

echo "genrate nb feature file" >/dev/stderr
NB_PREFIX="$OUT_PREFIX/nb"
mkdir -p $NB_PREFIX
python trans_in_nb_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 1 -o "${NB_PREFIX}/uni.nb"
python trans_in_nb_feature.py -ptrain $PTRAIN -ptest $PTEST -ntrain $NTRAIN -ntest $NTEST -g 2 -o "${NB_PREFIX}/unibi.nb"
