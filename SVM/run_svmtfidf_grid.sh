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
############## FOR IMDB , unigram
#BIAS="-1"
#W_POSITIVE="1"
#W_NEGATIVE="10"
#C="110,140,170,200"


########### FOR IMDB , bigram

#BIAS="-1"
#W_POSITIVE="1"
#W_NEGATIVE="12.5"
#C="400,450,500,550,600"

########## FOR IMDB , uni , undersampling
#BIAS="-1"
#W_POSITIVE="1"
#W_NEGATIVE="1"
#C="100,150,200,250,300,350,400,450,500"

######### FOR IMDB , uni+bi , undersampling

#BIAS="-1"
#W_POSITIVE="1"
#W_NEGATIVE="1"
#C="350,400,450,500,550,600"

######### FOR IMDB , uni , oversampling

BIAS="-1"
W_POSITIVE="1"
W_NEGATIVE="1"
C="4096,8192"
######### FOR IMDB , bi , oversampling

#BIAS="-1,1"
#W_POSITIVE="5,1,1"
#W_NEGATIVE="1,1,5"
#C="300,350,400,450,500,550,600,650,700,750,800,850,900,950"


python svmtfidf_grid.py -p $POS -n $NEG -g $GRAM -cv $CV_NUM -o $RST_F -b_r '"'$BIAS'"' -w_p '"'$W_POSITIVE'"' -w_n '"'$W_NEGATIVE'"' -c_r '"'$C'"' 
