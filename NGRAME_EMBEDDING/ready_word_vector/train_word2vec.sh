#/bin/sh
EN_Corpus="/users1/wxu/data/word2vector/en_source/1billion"
EN_OUTPUT="/users1/wxu/data/word2vector/en/1billion.wordvectors"
/users1/wxu/bin/word2vec/word2vec -train $EN_Corpus -output $EN_OUTPUT -threads 4
