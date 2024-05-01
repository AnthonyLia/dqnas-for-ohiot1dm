# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:04:35 2022

@author: AnshumaanChauhan
"""

import os
import warnings
import pandas as pd
import tensorflow
import tensorflow.keras
from tensorflow.keras.optimizers import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose, LSTM
from keras.layers import MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization
import CNNCONSTANTS
from CNNCONSTANTS import *
#from time_series import *
class CNNSearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()


    def vocab_dict(self):
        
        #---------------------------Hyperparameter pool selection----------------------------------#

        #For fully connected
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu', 'selu', 'swish']
        
        #For Convolutional Layers
        conv_layers=['conv1d','separableconv1d','depthwiseconv1d','conv1dtranspose']
        conv_filters=[16,32,64,96,128,160,192,224,256]
        #conv_filter_size=[3,5,7,9]
        conv_filter_size=[2,3,4,5]
        conv_stride=[1,2,3]
        #conv_padding= ['same','valid']
        conv_padding=['same']
        #conv_padding=['causal']
        conv_weight_initializers=['HeNormal','HeUniform','RandomNormal','RandomUniform']
        conv_bias_initializers=['HeNormal','HeUniform','RandomNormal','RandomUniform']
        conv_regularizers=['l1','l2','l1_l2']
        
        #For Pooling Layers
        pool_layers=['maxpool1d','avgpool1d','globalmaxpool1d','globalavgpool1d']
        pool_size=[2,3,4,5]
        pool_stride=[1,2,3,4,5]
        pool_padding=['same','valid']

        #RegularizationLayers
        #reg_layers=['dropout','spatialDropout','alphaDropout']
        reg_layers=['dropout']
        dropout_rate=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        #For LSTM Layers
        lstm_activation=act_funcs
        lstm_recurrent_activation=act_funcs
        lstm_kernel_initializer=['glorot_uniform','glorot_normal','RandomNormal','RandomUniform']
        lstm_recurrent_initializer=['orthogonal','RandomNormal','RandomUniform']
        lstm_dropout=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        lstm_recurrent_dropout=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
        #Learning rate (Not included in the Search Space, this we will use in the training part on our own, that's why semi-automatic)
        #self.lr=[0.1,0.2,0.3,0.4,0.5,0.6]
        self.lr=[0.1]
        self.batch_size=[2,4,8,16]
        #self.learning_optimimzer=['adam','rms','sgd']
        self.learning_optimimzer=['adam']

        # initialize lists for keys and values of the vocabulary
        layer_params = []
        layer_id = []
        
        #---------------Starting creation of Voacb from which Controller will create a sequence-------------------------#

        ind=1
        
        for a in conv_layers:
          for b in conv_filters:
            for c in conv_filter_size:
              for d in conv_stride:
                for e in conv_padding:
                  for f in conv_weight_initializers:
                    for g in conv_bias_initializers:
                      for h in conv_regularizers:
                        if a is 'depthwiseconv1d':
                            layer_params.append((a,c,d,e,f,g,h))
                            layer_id.append(ind)
                            ind=ind+1
                        else:
                            layer_params.append((a,b,c,d,e,f,g,h))
                            layer_id.append(ind)
                            ind=ind+1
        self.conv_id=ind-1
        
        for a in pool_layers:
          for b in pool_size:
            for c in pool_stride:
              for d in pool_padding:
                  if a=="globalavgpool1d" or a=="globalmaxpool1d":
                      layer_params.append((a))
                      layer_id.append(ind)
                      ind+=1
                  else:
                      layer_params.append((a,b,c,d))
                      layer_id.append(ind)
                      ind=ind+1
        self.pool_id=ind-1
        
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(ind)
                ind=ind+1
        self.fully_id=ind-1
        
        for a in reg_layers:
          for b in dropout_rate:
            layer_params.append((a,b))
            layer_id.append(ind)
            ind=ind+1
        self.reg_layer_id=ind-1
        
        for a in lstm_activation:
            for b in lstm_recurrent_activation:
                for c in lstm_kernel_initializer:
                    for d in lstm_recurrent_initializer:
                        for e in lstm_dropout:
                            for f in lstm_recurrent_dropout:
                                layer_params.append((a,b,c,d,e,f))
                                layer_id.append(ind)
                                ind=ind+1
        self.lstm_layer_id=ind-1
        
        
        # zip the id and configurations into a dictionary
        vocab = dict(zip(layer_id, layer_params))
        
        # add Flatten and BatchNormalization in the volcabulary
        vocab[len(vocab)+1] = (('Flatten'))
        vocab[len(vocab)+1] = (('BatchNormalization'))
        
        # add the final softmax/sigmoid layer in the vocabulary
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            #vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
            #vocab[len(vocab) + 1] = (self.target_classes, 'sigmoid')
            vocab[len(vocab) + 1] = (self.target_classes, 'linear')
        return vocab

#--------------------------------------------Search Space Created--------------------------------------------#
                        
	# function to encode a sequence of configuration tuples
    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence


	# function to decode a sequence back to configuration tuples
    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence
    
class CNNGenerator(CNNSearchSpace):

    def __init__(self):

        self.target_classes = TARGET_CLASSES
        self.mlp_decay= MLP_DECAY
        self.mlp_momentum= MLP_MOMENTUM
        self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.mlp_one_shot = MLP_ONE_SHOT
        #self.metrics = ['mean_absolute_error','mean_squared_error']
        #self.metrics = ['mean_absolute_error']
        #self.metrics = [tensorflow.keras.metrics.MeanAbsoluteError()]
        self.metrics = [tensorflow.keras.metrics.MeanAbsoluteError(),tensorflow.keras.metrics.RootMeanSquaredError()]
        
        super().__init__(TARGET_CLASSES)
        
        if self.mlp_one_shot:
	
            # path to shared weights file 
            self.weights_file = 'LOGS2/shared_weights.pkl'
            
            # open an empty dataframe with columns for bigrams IDs and weights
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
        
            # pickle the dataframe
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)
                
    # function to create a keras model given a sequence and input data shape
    def create_model(self, sequence, cnn_input_shape):

            # decode sequence to get nodes and activations of each layer
            layer_configs = self.decode_sequence(sequence)
            try:
                # create a sequential model
                model = Sequential()

                for i, layer_conf in enumerate(layer_configs):
                    if i==0:
                        #print('%s is input_shape' %str(cnn_input_shape))
                        if layer_conf[0] == 'conv1d':
                            model.add(Conv1D(filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],kernel_initializer=layer_conf[5], bias_initializer=layer_conf[6], kernel_regularizer=layer_conf[7], input_shape=cnn_input_shape))
                            continue
                        elif layer_conf[0] == 'separableconv1d':
                            model.add(SeparableConv1D(filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],depthwise_initializer=layer_conf[5], bias_initializer=layer_conf[6], depthwise_regularizer=layer_conf[7], input_shape=cnn_input_shape))
                            continue
                        elif layer_conf[0] == 'depthwiseconv1d':
                            model.add(DepthwiseConv1D(kernel_size=layer_conf[1],strides=layer_conf[2],padding=layer_conf[3],depthwise_initializer=layer_conf[4], bias_initializer=layer_conf[5], depthwise_regularizer=layer_conf[6], input_shape=cnn_input_shape))
                            continue
                        else:
                            model.add(Conv1DTranspose(filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],kernel_initializer=layer_conf[5], bias_initializer=layer_conf[6], kernel_regularizer=layer_conf[7], input_shape=cnn_input_shape))
                            continue
                    elif layer_conf[0] == 'conv1d':
                        model.add(Conv1D(activation='relu',filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],kernel_initializer=layer_conf[5], bias_initializer=layer_conf[6], kernel_regularizer=layer_conf[7]))#, input_shape=cnn_input_shape))
                    elif layer_conf[0] == 'separableconv1d':
                        model.add(SeparableConv1D(activation='relu',filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],depthwise_initializer=layer_conf[5], bias_initializer=layer_conf[6], depthwise_regularizer=layer_conf[7]))#, input_shape=cnn_input_shape))    
                    elif layer_conf[0] == 'depthwiseconv1d':
                        model.add(DepthwiseConv1D(activation='relu',kernel_size=layer_conf[1],strides=layer_conf[2],padding=layer_conf[3],depthwise_initializer=layer_conf[4], bias_initializer=layer_conf[5], depthwise_regularizer=layer_conf[6]))#, input_shape=cnn_input_shape))
                    elif layer_conf[0] == 'conv1dtranspose':
                        model.add(Conv1DTranspose(activation='relu',filters=layer_conf[1],kernel_size=layer_conf[2],strides=layer_conf[3],padding=layer_conf[4],kernel_initializer=layer_conf[5], bias_initializer=layer_conf[6], kernel_regularizer=layer_conf[7]))#, input_shape=cnn_input_shape))
                    elif layer_conf[0] == 'maxpool1d':
                        model.add(MaxPooling1D(activation='relu',pool_size=(layer_conf[1],layer_conf[1]),strides=(layer_conf[2],layer_conf[2]),padding=layer_conf[3]))  
                    elif layer_conf[0] == 'avgpool1d':
                        model.add(AveragePooling1D(activation='relu',pool_size=(layer_conf[1],layer_conf[1]),strides=(layer_conf[2],layer_conf[2]),padding=layer_conf[3]))
                    elif layer_conf[0] == 'globalmaxpool1d':
                        model.add(GlobalMaxPooling1D())
                    elif layer_conf[0] == 'globalavgpool1d':
                        model.add(GlobalAveragePooling1D())
                    elif layer_conf[0] in ['sigmoid', 'tanh', 'relu', 'elu', 'selu', 'swish']:
                        model.add(LSTM(32, activation=layer_conf[0], recurrent_activation=layer_conf[1], kernel_initializer=layer_conf[2], recurrent_initializer=layer_conf[3], dropout=layer_conf[4], recurrent_dropout=layer_conf[5], return_sequences=True ))
                    elif layer_conf == 'dropout':
                        model.add(Dropout(layer_conf[1], name='dropout'))
                    elif layer_conf == 'Flatten':
                        model.add(Flatten())
                    elif layer_conf == 'BatchNormalization':
                        model.add(BatchNormalization())
                    else:
                        model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
            
            #print(model.summary())
            # return the keras model
                return model
            except Exception as e:
                print(e)
                return None
            except ValueError:
                #print("-----------------------Received model that gets negative values for input image after processing------------------")
                return None
    
    # function to compile the model with the appropriate optimizer and loss function
    def compile_model(self, model):
        models=[]
        # Learning rate and Optimizer are changed and model is complied multiple times
            # compile model 
        
        for i in self.lr:
            for j in self.learning_optimimzer:
                """lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = i,
                    decay_steps = 10000,
                    decay_rate = 0.0)"""
                if j is 'sgd':
                    optim = tensorflow.keras.optimizers.legacy.SGD( lr = i, decay = self.mlp_decay )
                    #optim = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule)
                    #optim = tensorflow.keras.optimizers.SGD(lr=i, decay=self.mlp_decay, momentum=self.mlp_momentum)
                elif j is 'adam':
                    optim=tensorflow.keras.optimizers.legacy.Adam( lr = i, decay = self.mlp_decay )
                    #optim=tensorflow.keras.optimizers.Adam( learning_rate = lr_schedule )
                    #optim=tensorflow.keras.optimizers.Adam( lr=i, decay=self.mlp_decay)
                else:
                    optim=tensorflow.keras.optimizers.legacy.RMSprop(lr=i, decay=self.mlp_decay)
                    #optim=tensorflow.keras.optimizers.RMSprop(lr=i, decay=self.mlp_decay)
                '''if (tensorflow.test.gpu_device_name()):g
                    # Maybe try following
                    with tf.device("/gpu:0"):
                        model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
                else:
                    model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
                    
'''
                #model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)
                #model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), optimizer=optim, metrics=self.metrics)
                model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), optimizer=tensorflow.keras.optimizers.Adam(epsilon=1e-5), metrics=self.metrics)
                models.append(model)
            #------------------Always check whether the loss function and metrics is in accordance with the target classes and the dataset-------------#
            # return a list of compiled keras model
        
        #optim=tensorflow.keras.optimizers.Adam(lr=0.2,decay=self.mlp_decay)
        #model.compile(loss=self.mlp_loss_func,optimizer=optim,metrics=self.metrics)
        return models

        
    def set_model_weights(self, model):
        #print(model)
        # get nodes and activations for each layer 
        layer_configs=[]
        for layer in model.layers:
            #print(layer.name)
            #print(layer.get_config())
            # add flatten since it affects the size of the weights
            #index=layer.get_config()['name'].rfind("_")    
            if 'flatten' in layer.name:
                layer_configs.append((layer.input_shape,'Flatten'))
            # don't add dropout since it doesn't affect weight sizes or activations
            elif not (('dropout' in layer.name) or ('max_pooling1d' in layer.name) or ('average_pooling1d' in layer.name) or ('global_max_pooling1d' in layer.name) or ('global_average_pooling1d' in layer.name)):
                #For Conv Layers 
                if 'separable_conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'separable_conv1d',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                elif 'depthwise_conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'depthwise_conv1d',layer.get_config()['kernel_size']))

                elif 'conv1d_transpose' in layer.name:
                    layer_configs.append((layer.input_shape,'conv1d_transpose',layer.get_config()['filters'],layer.get_config()['kernel_size']))

                elif 'conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'conv1d',layer.get_config()['filters'],layer.get_config()['kernel_size']))
               
                #For BatchNormalization Layer
                elif layer.name is 'batch_normalization':
                    layer_configs.append((layer.input_shape,'batch_normalization',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                    #layer_configs.append(layer.get_config()['name'][:layer.get_config()['name'].index("_")],layer.get_config()['filters'],layer.get_config()['kernel_size'])
                
                #For Dense Layers
                else :
                    layer_configs.append((layer.input_shape,layer.get_config()['units'], layer.get_config()['activation']))
        
        # get bigrams of relevant layers for weights transfer
        config_ids = []
        #Starting from 1 as we are using i-1 in the saving part
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        # for all layers
        j = 0
        #print('---------------------------Setting Weights-------------------')
        for i, layer in enumerate(model.layers):
            if j >= len(config_ids):
                break
            if not (('dropout' in layer.name) or ('max_pooling1d' in layer.name) or ('average_pooling1d' in layer.name) or ('global_max_pooling1d' in layer.name) or ('global_average_pooling1d' in layer.name)):
                warnings.simplefilter(action='ignore', category=FutureWarning)
                
                
                # get all bigram values we already have weights for
                bigram_ids = self.shared_weights['bigram_id'].values
                #print("Layer : {0}------ Bigram :{1}".format(i,bigram_ids))
                # check if a bigram already exists in the dataframe
                search_index = []
                for x in range(len(bigram_ids)):
                    #print("C0: ",config_ids[j][0][1:])
                    #print("B0: ",bigram_ids[x][0][1:])
                    #print("C1: ",config_ids[j][1])
                    #print("B1: ",bigram_ids[x][1])
                    #print("Config value of first:",config_ids[j][0][0:])
                    #print("Bigram value of first:",bigram_ids[x][0][0:])
                    if ((config_ids[j][0][1:] == bigram_ids[x][0][1:]) and (config_ids[j][1]==bigram_ids[x][1])):
                        search_index.append(x)
                
                # set layer weights if there is a bigram match in the dataframe 
                if len(search_index) > 0:
                    #print("Transferring weights for layer:", config_ids[j])
                    try:
                        #print('transferring weights in set_model_weights')
                        layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                    except:
                        print(f"errror with set_weights. i in layers is {i} and layer is {layer}, self.shared_weights['weights'].values[search_index[0]] is {self.shared_weights['weights'].values[search_index[0]]}, and search_index is {search_index}.")
                j += 1
            
    def update_weights(self, model):

        # get nodes and activations for each layer
        layer_configs = []
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append((layer.input_shape,'Flatten'))
            elif not (('dropout' in layer.name) or ('max_pooling1d' in layer.name) or ('average_pooling1d' in layer.name) or ('global_max_pooling1d' in layer.name) or ('global_average_pooling1d' in layer.name)):
                if 'separable_conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'separable_conv1d',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                elif 'depthwise_conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'depthwise_conv1d',layer.get_config()['kernel_size']))
                elif 'conv1d_transpose' in layer.name:
                    layer_configs.append((layer.input_shape,'conv1d_transpose',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                elif 'conv1d' in layer.name:
                    layer_configs.append((layer.input_shape,'conv1d',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                elif layer.name is 'batch_normalization':
                    layer_configs.append((layer.input_shape,'batch_normalization',layer.get_config()['filters'],layer.get_config()['kernel_size']))
                else :
                    layer_configs.append((layer.input_shape,layer.get_config()['units'], layer.get_config()['activation']))
           
            
       
        # get bigrams of relevant layers for weights transfer
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        # for all layers
        j = 0
        #print('-------------------Updating weights--------------------------')
        for i, layer in enumerate(model.layers):
            if j >= len(config_ids):
                break
            if not (('dropout' in layer.name) or ('max_pooling1d' in layer.name) or ('average_pooling1d' in layer.name) or ('global_max_pooling1d' in layer.name) or ('global_average_pooling1d' in layer.name)):
                warnings.simplefilter(action='ignore', category=FutureWarning)
                
                #get all bigram values we already have weights for
                bigram_ids = self.shared_weights['bigram_id'].values
                #print("Layer : {0}------ Bigram :{1}".format(i,bigram_ids))
                # check if a bigram already exists in the dataframe
                search_index = []
                for x in range(len(bigram_ids)):
                    if ((config_ids[j][0][1:] == bigram_ids[x][0][1:]) and (config_ids[j][1]==bigram_ids[x][1])):
                        search_index.append(x)
                
                # add weights to df in a new row if weights aren't already available
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                      'weights': layer.get_weights()},
                                                                     ignore_index=True)
                # else update weights 
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)
    
    
    
    def train_model(self, models, window, nb_epochs, validation_split=0.1, callbacks=None):
        history_of_models=None
        mae = float('inf')
        if callbacks == None:
            callbacks = [
                tensorflow.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=3)
                ]
        for model in models:
            if self.mlp_one_shot:
                self.set_model_weights(model)
                for batch_size_value in self.batch_size:
                    history = model.fit(window.train,
                                epochs=nb_epochs,
                                validation_data=window.val,
                                callbacks=callbacks,
                                verbose=0)
                    if history.history['mean_absolute_error'][0] < mae :
                        mae=history.history['mean_absolute_error'][len(history.history['mean_absolute_error'])-1]
                        history_of_models=history
                        self.update_weights(model)
            else:
                for batch_size_value in self.batch_size:
                    history = model.fit(window.train,
                                epochs=nb_epochs,
                                batch_size=batch_size_value,
                                validation_data=window.val,
                                callbacks=callbacks,
                                verbose=0)
                    if history.history['mean_absolute_error'][0] < mae:
                        mae=history.history['mean_absolute_error'][len(history.history['mean_absolute_error'])-1]
                        history_of_models=history
        
        return history_of_models
