#/bin/sh

if [ $# != 4 ]; then
    echo "$0 [postrain] [negtrain] [postest] [negtest]" >> /dev/stderr 
    exit 1
fi
############# NEED to SPECIFY PARAMS FOR DIFFERENT METHOD #################
POSTRAIN="$1"
NEGTRAIN="$2"
POSTEST="$3"
NEGTEST="$4"

POUT_F="tmp.pout"

TMP_MODEL="tmp.model"
#echo "=================NBSVM , unigram  =========="
#python nbsvm_train.py --postrain $POSTRAIN --negtrain $NEGTRAIN --ngram 1 --out $TMP_MODEL -c 1 -w_p 1 -w_n 1 -b -1 -beta 0.25
#python nbsvm_test.py --postest $POSTEST --negtest $NEGTEST --model $TMP_MODEL --predict_f $POUT_F
#python analysis_result.py --rstf $POUT_F
#echo "=================NBSVM , unigram + bigram ================"
#python nbsvm_train.py --postrain $POSTRAIN --negtrain $NEGTRAIN --ngram 2 --out $TMP_MODEL -c 1 -w_p 1 -w_n 1 -b -1 -beta 0.25
#python nbsvm_test.py --postest $POSTEST --negtest $NEGTEST --model $TMP_MODEL --predict_f $POUT_F
#python analysis_result.py --rstf $POUT_F
#
#echo "================SVM(BOOL) , unigram ======================"
#python svmbool_test.py --postrain $POSTRAIN --negtrain $NEGTRAIN --postest $POSTEST --negtest $NEGTEST --ngram 1 --predict_f $POUT_F -c 0.1 -w_p 1 -w_n 1 -b -1
#python analysis_result.py --rstf $POUT_F
#
#echo "================SVM(BOOL) , unigram + bigram =============="
#python svmbool_test.py --postrain $POSTRAIN --negtrain $NEGTRAIN --postest $POSTEST --negtest $NEGTEST --ngram 2 --predict_f $POUT_F -c 0.1 -w_p 1 -w_n 1 -b -1
#python analysis_result.py --rstf $POUT_F

echo "================SVM(TFIDF) , unigram ====================="
python svmtfidf_test.py -ptrain $POSTRAIN -ptest $POSTEST -ntrain $NEGTRAIN -ntest $NEGTEST -g 1 -c 800 -w_p 1 -w_n 1 -b -1

echo "================SVM(TFIDF) , unigram + bigram ============"
python svmtfidf_test.py -ptrain $POSTRAIN -ptest $POSTEST -ntrain $NEGTRAIN -ntest $NEGTEST -g 2 -c 2100 -w_p 1 -w_n 1 -b -1

rm -f $TMP_MODEL $POUT_F
