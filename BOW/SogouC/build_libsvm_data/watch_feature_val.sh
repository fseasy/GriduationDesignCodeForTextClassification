#!/bin/sh
if [ $# -ne 2 ] ;then
    echo "usage: $0 [scale_feature_file] [line_num]" > /dev/stderr
    exit 0
fi
cat $1 | sed -n ''$2'p' | awk -F' ' '{for(i=1;i<=NF;i++){print $i ;}}' | awk -F':' '{if(NF==2){if($2 != "-1"){print $0}}}'
