#/bin/sh

if [ $# -ne 4 ] ;then
    echo 'usage : $0 [pos_corpus] [neg_corpus] [gram_num] [cv_result_f]' >/dev/stderr
    exit 1
fi
POS=$1
NEG=$2

#POS="data/postrain"
#NEG="data/negtrain"
GRAM=$3
OUT_F=$4
CV_NUM=5
########### NBSVM  UNIGRAM
BIAS="-1,1"
#W_POSITIVE="1,1,1"
#W_NEGATIVE="1,5,10"
#C="0.0001,0.0005,0.001,0.005,0.01"
#BETA="0,0.125,0.25"

########### NBSVM UNI + BIGRAM
#W_POSITIVE="1,1,1"
#W_NEGATIVE="10,15,20"
#C="0.00001,0.000025,0.00005,0.000075,0.0001"
#BETA="0.125"

########## NBSVM uni , under sampling

#W_POSITIVE="5,1,1,1,1"
#W_NEGATIVE="1,1,5,10,15"
#C="0.00001,0.0001,0.001,0.01,0.1,1,10"
#BETA="0.25,0.5,0.75"

######### NBSVM uni+bi , IMDB , undersampling

#W_POSITIVE="5,1,1,1,1"
#W_NEGATIVE="1,1,5,10,15"
#C="0.00001,0.0001,0.001,0.01,0.1,1,10"
#BETA="0.25,0.5,0.75"

######## NBSVM uni , IMDB , oversampling

#W_POSITIVE="5,1,1"
#W_NEGATIVE="1,1,5"
#C="0.00001,0.0001,0.001,0.01,0.1,1,10"
#BETA="0.25,0.5,0.75"

####### NBSVM uni+bi , IMDB , oversampling


W_POSITIVE="15,10,5"
W_NEGATIVE="1,1,1"
C="0.1"
BETA="0.5"

python nbsvm_grid.py -p $POS -n $NEG -g $GRAM -cv $CV_NUM -o $OUT_F -b_r '"'$BIAS'"' -w_p '"'$W_POSITIVE'"' -w_n '"'$W_NEGATIVE'"' -c_r '"'$C'"' -beta_r '"'$BETA'"'
