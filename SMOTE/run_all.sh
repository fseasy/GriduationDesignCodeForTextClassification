#/bin/sh

SMOTE_TRAIN_BOOL_UNI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/bool/uni.bool.smote.train"
SMOTE_TRAIN_BOOL_UNIBI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/bool/unibi.bool.smote.train"

SMOTE_TRAIN_NB_UNI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/nb/uni.nb.smote.train"
SMOTE_TRAIN_NB_UNIBI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/nb/unibi.nb.smote.train"

SMOTE_TRAIN_TFIDF_UNI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/tfidf/uni.tfidf.smote.train"
SMOTE_TRAIN_TFIDF_UNIBI="/users1/wxu/code/text_category/SMOTE/smote_result/imdb/tfidf/unibi.tfidf.smote.train"

[ -e $SMOTE_TRAIN_BOOL_UNI ] && [ -e $SMOTE_TRAIN_BOOL_UNIBI ] && [ -e $SMOTE_TRAIN_NB_UNI ] && [ -e $SMOTE_TRAIN_NB_UNIBI ] &&
[ -e $SMOTE_TRAIN_TFIDF_UNI ] && [ -e $SMOTE_TRAIN_TFIDF_UNIBI ] && echo "data set check ok" >/dev/stderr


TEST_BOOL_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/bool/uni.bool.test"
TEST_BOOL_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/bool/unibi.bool.test"

TEST_NB_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/nb/uni.nb.test"
TEST_NB_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/nb/unibi.nb.test"

TEST_TFIDF_UNI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/tfidf/uni.tfidf.test"
TEST_TFIDF_UNIBI="/users1/wxu/code/text_category/SMOTE/data/imdb.10vs1/tfidf/unibi.tfidf.test"

echo "================SVM BOOL UNI==============="
python svm_classify.py -train $SMOTE_TRAIN_BOOL_UNI -test $TEST_BOOL_UNI -w_p 1 -w_n 1 -b -1 -c 0.1
echo "================SVM BOOL UNIBI==============="
python svm_classify.py -train $SMOTE_TRAIN_BOOL_UNIBI -test $TEST_BOOL_UNIBI -w_p 1 -w_n 1 -b -1 -c 0.1

echo "================SVM NB UNI==============="
python nbsvm_classify.py -train $SMOTE_TRAIN_NB_UNI -test $TEST_NB_UNI -w_p 1 -w_n 1 -b -1 -c 1 -beta 0.25 
echo "================SVM NB UNIBI==============="
python nbsvm_classify.py -train $SMOTE_TRAIN_NB_UNIBI -test $TEST_NB_UNIBI -w_p 1 -w_n 1 -b -1 -c 1 -beta 0.25

echo "================SVM TFIDF UNI==============="
python svm_classify.py -train $SMOTE_TRAIN_TFIDF_UNI -test $TEST_TFIDF_UNI -w_p 1 -w_n 1 -b -1 -c 800
echo "================SVM TFIDF UNIBI==============="
python svm_classify.py -train $SMOTE_TRAIN_TFIDF_UNIBI -test $TEST_TFIDF_UNIBI -w_p 1 -w_n 1 -b -1 -c 2100


echo "=================BOOSTING BOOL UNI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_BOOL_UNI -test $TEST_BOOL_UNI
echo "=================BOOSTING BOOL UNIBI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_BOOL_UNIBI -test $TEST_BOOL_UNIBI

echo "=================BOOSTING NB UNI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_NB_UNI -test $TEST_NB_UNI
echo "=================BOOSTING NB UNIBI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_NB_UNIBI -test $TEST_NB_UNIBI

echo "=================BOOSTING TFIDF UNI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_TFIDF_UNI -test $TEST_TFIDF_UNI
echo "=================BOOSTING TFIDF UNIBI==============="
python /users1/wxu/code/text_category/BOOSTEDTREE/xgboost_classify.py -train $SMOTE_TRAIN_TFIDF_UNIBI -test $TEST_TFIDF_UNIBI

