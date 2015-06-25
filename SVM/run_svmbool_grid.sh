#/bin/sh

if [ $# -ne 4 ] ;then
    echo 'usage : $0 [pos_corpus] [neg_corpus] [gram_n] [result_f]' >/dev/stderr
    exit 1
fi
POS=$1
NEG=$2

#POS="data/postrain"
#NEG="data/negtrain"
GRAM=$3
RST_F=$4
CV_NUM=5

######### FOR unigram , IMDB
BIAS="-1,1"
#W_POSITIVE="1,1,1"
#W_NEGATIVE="1,5,10" W_N = 5,10 good
#W_POSITIVE="1,1,1"
#W_NEGATIVE="5,7.5,10"
#C="0.0001,0.001,0.01,0.1,0.5,1" # 0.001 ~ 0.01 good
#C="0.001,0.0025,0.005,0.0075,0.01"


############# FOR IMDB unigram + bigram
#W_POSITIVE="1,1,1,1"
#W_NEGATIVE="15,17.5,20,22.5"
#C="0.00075"


############# FOR IMDB uni , undersampling

#W_POSITIVE="5,1,1,1"
#W_NEGATIVE="1,1,5,10"
#C="0.0001,0.001,0.01,0.1,1,10"

############ FOR IMDB uni+bi , undersampling


#W_POSITIVE="5,1,1,1"
#W_NEGATIVE="1,1,5,10"
#C="0.0001,0.001,0.01,0.1,1,10"

########### FOR IMDB , uni , oversampling

#W_POSITIVE="11,14,17,20"
#W_NEGATIVE="1,1,1,1"
#C="0.01"

########### FOR IMDB , uni , oversampling
W_POSITIVE="25,20,15"
W_NEGATIVE="1,1,1"
C="0.001"

python svmbool_grid.py -p $POS -n $NEG -g $GRAM -cv $CV_NUM -o $RST_F -b_r '"'$BIAS'"' -w_p '"'$W_POSITIVE'"' -w_n '"'$W_NEGATIVE'"' -c_r '"'$C'"' 
