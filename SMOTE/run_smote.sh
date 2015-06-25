#/bin/sh

IMDB_BOOL_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/bool/uni.bool.train"
IMDB_BOOL_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/bool/unibi.bool.train"

IMDB_TFIDF_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/tfidf/uni.tfidf.train"
IMDB_TFIDF_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/tfidf/unibi.tfidf.train"

IMDB_NB_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/nb/uni.nb.train"
IMDB_NB_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/nb/unibi.nb.train"

PRE_PATH="smote_result/imdb"

BOOL_PATH="$PRE_PATH/bool"
mkdir -p $BOOL_PATH
echo "SMOTE IMDB BOOL" >/dev/stderr
#python smote_data.py -d $IMDB_BOOL_UNI -s 0 -r 9 -o "$BOOL_PATH/uni.bool.smote.train"
#python smote_data.py -d $IMDB_BOOL_UNIBI -s 0 -r 9 -o "$BOOL_PATH/unibi.bool.smote.train"

TFIDF_PATH="$PRE_PATH/tfidf"
mkdir -p $TFIDF_PATH
echo "SMOTE IMDB TFIDF" >/dev/stderr
python smote_data.py -d $IMDB_TFIDF_UNI -s 0 -r 9 -o "$TFIDF_PATH/uni.tfidf.smote.train"
#python smote_data.py -d $IMDB_TFIDF_UNIBI -s 0 -r 9 -o "$TFIDF_PATH/unibi.tfidf.smote.train"

NB_PATH="$PRE_PATH/nb"
mkdir -p $NB_PATH
echo "SMOTE IMDB NB" >/dev/stderr
python smote_data.py -d $IMDB_NB_UNI -s 0 -r 9 -o "$NB_PATH/uni.nb.smote.train"
#python smote_data.py -d $IMDB_NB_UNIBI -s 0 -r 9 -o "$NB_PATH/unibi.nb.smote.train"

