#/bin/sh

if [ $# -ne 3 ] ; then
    echo "usage: $0 [train_data] [test_data] [best c]" >/dev/stderr
    exit 1
fi
DIR_NAME="`dirname $1`"
RST_PATH="${DIR_NAME}/predict.result"
TMP_MODEL_F="tmp.model"
train -s 1 -c $3 $1 $TMP_MODEL_F
predict $2 $TMP_MODEL_F $RST_PATH

TY_F="${DIR_NAME}/true.result"
cat $2 | awk '{print $1}' > $TY_F
python calc_prf_in_semantic_concept.py -tf $TY_F -pf $RST_PATH

rm -f $TMP_MODEL_F
