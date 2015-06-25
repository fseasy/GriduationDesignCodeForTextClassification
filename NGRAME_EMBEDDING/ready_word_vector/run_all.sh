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
TMP_FILE="`date +%Y%m%d%H%S`.tmp"
sh run_build_dict_and_trans.sh $TYPE $gram | tee $TMP_FILE
dict_path="`tail -1 $TMP_FILE | awk '{print $NF}'`"
dict_name="`basename $dict_path`"
pre_name="${dict_name%%.*}"
echo $dict_path
echo $pre_name
sh run_cluster_dictvector_using_sofiaml.sh $dict_path $pre_name

rm $TMP_FILE
