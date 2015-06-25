#/bin/sh

python conv_from_chars.py -train_p ../NBSVM/data/imdb/train/postrain.imdb -train_n ../NBSVM/data/imdb/train/negtrain.imdb -test_p ../NBSVM/data/imdb/test/postest.imdb -test_n ../NBSVM/data/imdb/test/negtest.imdb -epochs 5
