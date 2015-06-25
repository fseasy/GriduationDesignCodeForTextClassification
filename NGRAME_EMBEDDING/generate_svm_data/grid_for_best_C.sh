#/bin/sh
if [ $# -ne 1 ]; then
    echo "usage: $0 [train_file]" >/dev/stderr
    echo 'should set a good range of C' >/dev/stderr
    exit 1
fi
echo "using grid.py to search for best C . should set a good range of C and step" >/dev/stderr
 ~/bin/libsvm-3.20/tools/grid.py -log2c -5,2,1 -log2g null -train /users1/wxu/bin/liblinear-1.96/train $1

