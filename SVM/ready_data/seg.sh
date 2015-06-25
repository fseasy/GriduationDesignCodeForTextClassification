#!/bin/sh
if [ $# -ne 2 ]; then
    echo "usage : $0 [indir] [outdir]" > /dev/stderr
    exit 0
fi
indir=$1
outdir=$2
if [ ! -d "$indir" ]; then
    echo "$indir does not exists" >/dev/stderr
    exit 0
fi
echo "input dir : $indir"
if [ ! -d "$outdir" ];then
    echo "output dir '""$outdir""'does not exists\n create it ? [y/n]"
    read r
    if [ "$r" == "y" ] ; then
        mkdir "$outdir"
    else
        exit 0
    fi
fi
echo "output dir : $outdir"
### do seg
f_l="`ls $indir`"
while read fname
do
    inpath="$indir/$fname"
    echo -n "seg $inpath"
    outpath="$outdir/${fname}_seged"
    echo " to $outpath "
    ## do seg 
    cat $inpath | python transcode.py --stream | /users1/exe/projects/ltp/bin/examples/cws_cmdline /data/ltp/ltp-models/3.2.0-server/ltp_data/cws.model > $outpath
done <<LIST
$f_l
LIST
