#/bin/sh 
PREPATH="cluster_result"
help()
{
cat << HELP

cluster the ngram using word2vec and sofia-ml 
saving result autoly to dir "$PREPATH"

usage : $0 [dict_data] [save_sub_path]
    
[dict_data] : path to the output of "run_build_dict_and_trans.sh" . file_name should like XXXX.xxx , using XXXX for saved file name
[save_sub_path] : output path = ${PREPATH}/${save_sub_path}/XXXX.***
HELP
    exit 0
}

[ ! -f $1 ] && help
[ $# -ne 2 ] && help

dict_data_name="`basename $1`"
dict_data_pre_name="${dict_data_name%%.*}"
saved_file_prepath="$PREPATH/$2/$dict_data_pre_name"
out_dir=`dirname $saved_file_prepath`

cat >/dev/stderr <<INFO2
dict_data path "$1"
saved datas path "${saved_file_prepath}.*"
INFO2
K_VAL=200
mkdir -p $out_dir
DICT_DATA=$1
OUT_F="${saved_file_prepath}.vectors.svm_k$K_VAL"
CLUSTER="${saved_file_prepath}.cluster.centroid_k$K_VAL"
ASSIGNMENT="${saved_file_prepath}.assignments_k$K_VAL"


python ready_sofia_ml.py --dictpath $DICT_DATA --out $OUT_F
/users1/wxu/bin/sofia-ml/sofia-ml/sofia-kmeans --k $K_VAL --init_type random --opt_type mini_batch_kmeans --mini_batch_size 100 --iterations 1000 --objective_after_init --objective_after_training --training_file $OUT_F --model_out $CLUSTER --dimensionality 140

/users1/wxu/bin/sofia-ml/sofia-ml/sofia-kmeans --model_in $CLUSTER --test_file $OUT_F --objective_on_test --cluster_assignments_out $ASSIGNMENT  

cat >/dev/stderr <<INFO
.....
out vectors saved at "$OUT_F"
cluster centroids saved at "$CLUSTER"
ngram assignment info saved at "$ASSIGNMENT"
INFO
