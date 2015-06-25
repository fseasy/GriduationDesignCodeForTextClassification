#/usr/bin/env python
#coding=utf8

import argparse
import logging
import time
from collections import defaultdict , OrderedDict

import numpy
import theano
import theano.tensor as T

from conv_net_classes import *
from load_data import *
from conv_config import *
from evaluate import *

logging.basicConfig(level=logging.INFO)

class ConvNet_Zhang(object) :
    def __init__(self , batch_size ) :
        self.batch_size = batch_size
    
    def ready_data(self , train_f_pos , train_f_neg , test_f_pos , test_f_neg) :
        logging.info('Try to load data ...')
        train_data_x_p , train_data_y_p = trans_data(train_f_pos , POSITIVE_LABEL) # In fact , data is just indices
        train_data_x_n , train_data_y_n = trans_data(train_f_neg , NEGATIVE_LABEL)
        train_data_x = numpy.vstack((train_data_x_p , train_data_x_n))
        train_data_y = numpy.hstack((train_data_y_p , train_data_y_n)) # Row array
        
        test_data_x_p , test_data_y_p = trans_data(test_f_pos , POSITIVE_LABEL)
        test_data_x_n , test_data_y_n = trans_data(test_f_neg , NEGATIVE_LABEL)
        test_data_x = numpy.vstack((test_data_x_p , test_data_x_n))
        test_data_y = numpy.hstack((test_data_y_p , test_data_y_n))
        
        train_data_x , train_data_y = self.replicate_enough_data_for_minibatch(train_data_x , train_data_y)
        test_data_x , test_data_y = self.replicate_enough_data_for_minibatch(test_data_x , test_data_y)
        
        self.train_data_instance_num = train_data_x.shape[0]
        self.test_data_instance_num = test_data_x.shape[0]

        self.train_data_x , self.train_data_y = self.shared_dataset(train_data_x , train_data_y)
        self.test_data_x , self.test_data_y = self.shared_dataset(test_data_x , test_data_y)
        
        logging.info('Loaded . trainning data instance %d , testing data instance %d ' %(self.train_data_instance_num , 
                                                                                         self.test_data_instance_num))


    def replicate_enough_data_for_minibatch(self , data_x , data_y) :
        '''
        type : numpy.ndarray
        make sure the data.shape[0] is the multiple of batch_sizes 
        '''
        assert( data_x.shape[0] == data_y.shape[0])
        numpy.random.seed(1234)
        instance_num = data_x.shape[0]
        
        needed_num = self.batch_size - instance_num % self.batch_size 
        if needed_num == self.batch_size : return 
        total_random_index = numpy.random.permutation(instance_num)
        while len(total_random_index) < self.batch_size :
            total_random_index = numpy.append(total_random_index , numpy.random.permutation(instance_num)) # add this because instance num may less than batch size !

        random_idx = total_random_index[:needed_num]
        data_x = numpy.append(data_x , data_x[random_idx , :] , axis=0 )
        data_y = numpy.append(data_y , data_y[random_idx])
        return data_x , data_y 
    
    def shared_dataset(self , data_x , data_y , borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
        dtype=theano.config.floatX),
        borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
        dtype=theano.config.floatX),
        borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    def build_model(self) :
        self.conv_layers = []
        
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        ## Assert self.x.shape[0] = batch_size , self.x.shape[1] = LINE_LEN_LIMIT
        
        ### Conv-Pool Layer 
        logging.info('Try to build Conv_Pool Layer .')
        
        init_n_input_feature_map = 1

        self.input_layer0 = ALPHABET_SHARED_VECTOR[T.cast(self.x.flatten() , dtype='int32')].reshape( (self.batch_size , init_n_input_feature_map , LINE_LEN_LIMIT , ALPHABET_LEN) )
        rng = numpy.random.RandomState(3435)
        input_img = self.input_layer0
        self.convpool_layers = []
        previous_layer_n_feature_map = init_n_input_feature_map
        conv_layer_counter = 1
        input_img_height = LINE_LEN_LIMIT
        input_img_width = ALPHABET_LEN
        filter_width = ALPHABET_LEN
        for conv_conf in CONV_CONF :
            filter_height = conv_conf['filter_height']
            filter_num = conv_conf['filter_num']
            pool_height = conv_conf['pool_height']
            
            filter_shape = (filter_num , previous_layer_n_feature_map , filter_height , filter_width)
            pool_size = (pool_height , 1)
            image_shape = (self.batch_size , previous_layer_n_feature_map , input_img_height , input_img_width )
            
            convpool_layer = LeNetConvPoolLayer(rng, input=input_img,image_shape=image_shape,
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=CONV_ACTIVATION)
            self.convpool_layers.append(convpool_layer)
            input_img = convpool_layer.output
            previous_layer_n_feature_map = filter_num
            input_img_height = ( input_img_height - filter_height + 1 ) / pool_height 
            input_img_width = 1
            filter_width = 1
            logging.info('build conv layer %d done . input_shape : %s , filter_shape: %s , pool_size : %s , output_image_shape : %s .' %( 
                         conv_layer_counter , image_shape , filter_shape , pool_size , 
                         (self.batch_size , previous_layer_n_feature_map , input_img_height , input_img_width)))
            conv_layer_counter += 1

        ### HiddenLayer
        logging.info('Try to build nn layer')
        self.hidden_layer0_input = input_img.flatten(2)
        input_img_size = input_img_width * input_img_height * previous_layer_n_feature_map
        self.nn = MLPDropout(rng=rng , input=self.hidden_layer0_input , input_layer_size=input_img_size ,
                             hidden_layer_sizes=NN_CONF['hidden_layer_sizes'] , 
                             output_layer_size=NN_CONF['output_layer_size'] , 
                             dropout_rates=NN_CONF['dropout_rates'] , 
                             hidden_layer_activation=NN_CONF['hidden_layer_activation'] )
        
        logging.info('Try to build gradients')

        self.params = self.nn.params + [ param for conv in self.convpool_layers for param in conv.params]
        self.cost = self.nn.negative_log_likelihood(self.y)
        self.dropout_cost = self.nn.dropout_negative_log_likelihood(self.y)
        self.grad_updates = self.sgd_updates_adadelta(self.params , self.dropout_cost , LR_DECAY , 1e-6 , SQR_NORM_LIMIT )
       
        logging.info('Try to build train model function')

        self.index = T.lscalar()
        self.train_model = theano.function(inputs=[self.index] , 
                                           outputs=self.cost ,
                                           updates=self.grad_updates ,
                                           givens={
                                               self.x : self.train_data_x[self.index * self.batch_size : (self.index + 1 ) * self.batch_size] ,
                                               self.y : self.train_data_y[self.index * self.batch_size : (self.index + 1 ) * self.batch_size]
                                           }
                                           )
        logging.info('Try to build test model function')

        self.get_vali_results = theano.function(inputs=[self.index] ,
                                          outputs=self.nn.get_results(self.y) ,
                                          givens={
                                                self.x:self.train_data_x[self.index * self.batch_size : (self.index + 1) * self.batch_size] ,
                                                self.y:self.train_data_y[self.index * self.batch_size : (self.index + 1) * self.batch_size]
                                          })
        self.get_test_results = theano.function(inputs=[self.index] ,
                                          outputs=self.nn.get_results(self.y) ,
                                          givens={
                                                self.x:self.test_data_x[self.index * self.batch_size : (self.index + 1) * self.batch_size] ,
                                                self.y:self.test_data_y[self.index * self.batch_size : (self.index + 1) * self.batch_size]
                                          })
        
    def sgd_updates_adadelta(self , params , cost , rho=0.95 , epsilon=1e-6 , norm_lim=9):
        """
        adadelta update rule, mostly from
        https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
        """
        def as_floatX(variable):
             if isinstance(variable, float):
                 return numpy.cast[theano.config.floatX](variable)
             if isinstance(variable, numpy.ndarray):
                 return numpy.cast[theano.config.floatX](variable)
             return theano.tensor.cast(variable, theano.config.floatX)
        updates = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        gparams = []
        for param in params:
            empty = numpy.zeros_like(param.get_value())
            exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
            gp = T.grad(cost, param)
            exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
            gparams.append(gp)
        for param, gp in zip(params, gparams):
            exp_sg = exp_sqr_grads[param]
            exp_su = exp_sqr_ups[param]
            up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
            updates[exp_sg] = up_exp_sg
            step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
            updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
            stepped_param = param + step
            if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param      
        return updates 
    def do_train_model(self , epochs , cv_num=10) :
        logging.info('Start to train model .')
        
        n_train_minibatchs = self.train_data_instance_num / self.batch_size
        print 'n_train_minibatchs : %d' %(n_train_minibatchs)
        cv_num = min(cv_num , n_train_minibatchs)
        for epoch in range(1 , epochs + 1) :
            # Once cn_num-fold Cross Validation , at the CV over , Get a Validate Error score or PRF score
            shuffled_minibatch_idx = numpy.random.permutation(n_train_minibatchs)
            n_cv_piece = n_train_minibatchs / cv_num 
            assert n_cv_piece >= 1
            ty , py = [] , []
            start_time = time.time()
            for cv_count in range(cv_num) :
                shuffled_vali_minibatch_idx_start = n_cv_piece * cv_count
                shuffled_vali_minibatch_idx_end = (cv_count + 1 ) * n_cv_piece if cv_count != cv_num -1 else n_train_minibatchs 
                shuffled_vali_minibatch_idx = range(shuffled_vali_minibatch_idx_start , shuffled_vali_minibatch_idx_end)
                shuffled_train_minibatch_idx = ( range(0 , shuffled_vali_minibatch_idx_start) + 
                                                 range(shuffled_vali_minibatch_idx_end , n_train_minibatchs) )

                cv_train_minibatch_idx = shuffled_minibatch_idx[shuffled_train_minibatch_idx]
                cv_vali_minibatch_idx = shuffled_minibatch_idx[shuffled_vali_minibatch_idx]
                # Train and Test For one CV
                logging.info('epoch : %d/%d , cv : %d/%d ' %(epoch , epochs , cv_count + 1 , cv_num))
                for minibatch_idx in cv_train_minibatch_idx :
                    self.train_model(minibatch_idx)
                for minibatch_idx in cv_vali_minibatch_idx :
                    ty_m , py_m = self.get_vali_results(minibatch_idx)
                    ty.extend(ty_m)
                    py.extend(py_m)
                logging.info('done')
            p_prf , n_prf = calc_prf(ty,py)
            end_time=time.time()
            logging.info('cv_results : P_class : %s , N_class : %s . Time Costing : %f mins' %(p_prf , n_prf , 
                          (end_time - start_time)/60 ))
        logging.info('Test Reulst ...')
        ty = []
        py = []
        for minibatch in range(self.test_data_instance_num / self.batch_size) :
            ty_m , py_m = self.get_test_results(minibatch)
            ty.extend(ty_m)
            py.extend(py_m)
        p_prf , n_prf = calc_prf(ty,py)
        logging.info('Test results : Positive Class :  %s , Negative Class : %s' %(p_prf , n_prf))
         
    def do_train_model_no_cv(self , epochs) :
        logging.info('Start to train model .')
        
        n_train_minibatchs = self.train_data_instance_num / self.batch_size
        print 'n_train_minibatchs : %d' %(n_train_minibatchs)
        for epoch in range(epochs) :
            for minibatch_idx in range(n_train_minibatchs) :
                self.train_model(minibatch_idx)
            py = []
            ty = []
            for minibatch_idx in range(n_train_minibatchs) :
                ty_m , py_m = self.get_vali_results(minibatch_idx)
                ty.extend(ty_m)
                py.extend(py_m)
            p_prf , n_prf = calc_prf(ty , py)
            logging.info('Valid Reuslt : P_class : %s . N_class : %s' %(p_prf , n_prf))
        n_test_minibatchs = self.test_data_instance_num / self.batch_size
        py = []
        ty = []
        for minibatch_idx in range(n_test_minibatchs) :
            ty_m , py_m = self.get_test_results(minibatch_idx)
            ty.extend(ty_m)
            py.extend(py_m)
        p_prf , n_prf = calc_prf(ty , py)
        logging.info('Test Result : P_class : %s . N_class : %s' %(p_prf , n_prf))
        print 'ty'
        print ' '.join(map(str , ty))
        print 'py'
        print ' '.join(map(str ,py))

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="ConvNet of Xiang Zhang")
    argp.add_argument('-train_p' , help="path to positive training data" , type=argparse.FileType('r') , default="sampledata/train_p")
    argp.add_argument('-train_n' , help="path to negative training data" , type=argparse.FileType('r') , default="sampledata/train_n")
    argp.add_argument('-test_p' , help="path to positive testing data" , type=argparse.FileType('r') , default="sampledata/test_p")
    argp.add_argument('-test_n' , help="path to negative testing data" , type=argparse.FileType('r') , default='sampledata/test_n')
    argp.add_argument('-epochs' , help="epoches to Train" , type=int , default='10')
    args = argp.parse_args()

    conv = ConvNet_Zhang(batch_size=BATCH_SIZE)
    conv.ready_data(args.train_p , args.train_n , args.test_p , args.test_n)
    conv.build_model()
    #conv.do_train_model(args.epochs , 5)
    conv.do_train_model_no_cv(args.epochs)
    args.train_p.close()
    args.train_n.close()
    args.test_p.close()
    args.test_n.close()
