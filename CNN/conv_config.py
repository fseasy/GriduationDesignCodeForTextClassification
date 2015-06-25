#/usr/bin/env python
#coding=utf8

CONV_CONF = [{'filter_height': 7 , 'filter_num' : 256 , 'pool_height' : 3 } ,
             {'filter_height': 7 , 'filter_num' : 256 , 'pool_height' : 3 } ,
             {'filter_height': 3 , 'filter_num' : 256 , 'pool_height' : 1 } ,
             {'filter_height': 3 , 'filter_num' : 256 , 'pool_height' : 1 } ,
             {'filter_height': 3 , 'filter_num' : 256 , 'pool_height' : 1 } ,
             {'filter_height': 3 , 'filter_num' : 256 , 'pool_height' : 3 } ,
            ]
CONV_ACTIVATION = "relu"

NN_CONF = { 'hidden_layer_sizes' : [1024,1024] ,
            'output_layer_size' : 2 ,
            'dropout_rates' : [ 0.5 , .5 ] ,
            'hidden_layer_activation' : 'relu' ,
            'lr_activation' : 'sigmoid'
          }

LR_DECAY = 0.95
SQR_NORM_LIMIT = 9
BATCH_SIZE = 128
