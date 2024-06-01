# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:41:33 2022

@author: AnshumaanChauhan
"""
'''
Edited by Anthony Liardo
'''

import CNNCONSTANTS
import pickle
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from NASutils import *
from CNNCONSTANTS import *
from DQNController import DQNAgent
from CNNGenerator import CNNGenerator
import matplotlib.pyplot as plt
from time_series2 import *
import gc
class CNNNAS(DQNAgent):


    def __init__(self, window):

        #self.x = x
        #self.y = y
        self.window = window
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.patient_data = {}
        self.data = []
        self.nas_data_log = 'LOGS2/nas_data.pkl'
        clean_log()

        super().__init__()

        self.CNNGenerator = CNNGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1)

        if self.use_predictor:
            self.controller_model = self.create_hybrid_model(self.controller_input_shape, self.controller_batch_size)
            self.target_model= self.create_hybrid_model(self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.create_control_model(self.controller_input_shape, self.controller_batch_size)
            self.target_model= self.create_control_model(self.controller_input_shape, self.controller_batch_size)

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.CNNGenerator.loss_func = 'binary_crossentropy'
        #model = self.CNNGenerator.create_model(sequence, (self.window.total_window_size-1,1))
        model = self.CNNGenerator.create_model(sequence, (6,1))
        #models = self.CNNGenerator.compile_model(model)
        if model==None:
            return model
        models = self.CNNGenerator.compile_model(model)
        models[0].summary()
        return models

    def train_architecture(self, model):
        #x, y = unison_shuffled_copies(self.x, self.y)
        #Check how to train models on different number of epochs 
        history_of_models = self.CNNGenerator.train_model(model, self.window, self.architecture_train_epochs)
        return history_of_models

    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        if len(history.history['mean_absolute_error']) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history['mean_absolute_error'][0],
                                  pred_accuracy])
                print('predicted accuracy: ',pred_accuracy)
            else:
                self.data.append([sequence,
                                  history.history['mean_absolute_error'][0]])
            print('mean absolute error: ', history.history['mean_absolute_error'][0])
        else:
            mae = np.ma.average(history.history['mean_absolute_error'],
                                    weights=np.arange(1, len(history.history['mean_absolute_error']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  mae,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  mae])
            print('Training MAE: ', mae)

    def prepare_controller_data(self, sequences):
        #Adds 0 at the end if the sequence length is shorter than Max length architecture 
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        #Have all the layers except the final softmax layer
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        #Final layer
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        #Getting val accuracy of the sequences
        mae_target = [item[1] for item in self.data]
        print(f'for reference, xc is {xc} and yc is {yc}, while sequences is {sequences}.')
        return xc, yc, mae_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, y_true, y_pred):
        baseline = 0.95
        reward = np.array([-item[1] + baseline if item[1] >= 0 else 0 for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        loss = - 0.08*K.log(y_pred) * discounted_reward[:, None] #+ 0.1*tf.math.sqrt(y_pred)
        #print(f"y_pred is {y_pred.numpy()} and y_true is {y_true.numpy()}")
        #print(f"reward is {reward}")
        for item in loss:
            #print(item)
            pass
        #print(f"discounted_reward is {discounted_reward} and loss is {loss.numpy()[0:3]}")
        return loss

    def train_controller(self, model, x, y, mae_amount, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model, 
                                    self.target_model,
                                    x,
                                    y,
                                    mae_amount,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     self.target_model,
                                     x,
                                     y,
                                     mae_amount,
                                     self.custom_loss,
                                     len(self.data)//2,
                                     self.controller_train_epochs)

    def search(self,current_patient,current_window):
        self.best_mae = 999
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
                print("At start print acc: ",pred_accuracies)
            for i, sequence in enumerate(sequences):
                print('New Architecture: ', self.decode_sequence(sequence))
                print('i is %d and sequence is %s' %(i,str(sequence)))
                
                model = self.create_architecture(sequence)
                if model==None:
                    if self.use_predictor:
                        self.data.append([sequence, 10.0, pred_accuracies[i]])
                        print('validation accuracy: ', 10.0)
                    else:
                        self.data.append([sequence, 10.0])
                        print('validation accuracy: ', 10.0)
                    print('------------------------------------------------------')
                    continue
                history = self.train_architecture(model)
                try:
                    print(f'history.history is {history.history}')
                except Exception as e:
                    self.data.append([sequence,10.0])
                    continue

                all_models = [('current_model',model[0]),
                              ]
                model_forecasts = {'current_model': [],
                    }
                #forecast_data = train_df[-6:].append(test_df[:-1]).values
                forecast_data = current_window.test_df[:-1].values
                N = len(current_window.test_df.values) - FORECAST_MINUTES//5 - 6
                predictions_array = np.zeros((len(current_window.test_df.values)-FORECAST_MINUTES//5-6,1))
                '''for i in range(len(test_df.values) - FORECAST_MINUTES//5-6):
                    pseudo_ground_truth = np.zeros((FORECAST_MINUTES//5 - 1,1))
                    current_window = forecast_data[i:i+6]
                    for j in range(FORECAST_MINUTES//5-1):
                        predict_window = np.append(current_window[j if j<6 else 6:],pseudo_ground_truth[:j])[None,:,None]
                        #print(predict_window)
                        #print(predict_window.shape)
                        
                        pseudo_ground_truth[j] = model[0].predict(predict_window,verbose=0)
                        
                    predict_window = np.append(current_window[j+1 if j+1<6 else 6:],pseudo_ground_truth[:j+1])[None,:,None]
                    predictions_array[i] = model[0].predict(predict_window,verbose=0)
                    if i % 50 == 0:
                        del pseudo_ground_truth
                        del current_window
                        del predict_window
                        gc.collect()
                    

                mae_ = np.abs( forecast_data[5+FORECAST_MINUTES//5:5+FORECAST_MINUTES//5+len(predictions_array)] - predictions_array).sum() / N
                print('Prediction MAE:',round(mae_,3))'''
                temp_eval = model[0].evaluate(self.window.test, verbose=0)
                child_mae = temp_eval[1]
                child_rmse = temp_eval[2]
                print('Prediction MAE:',child_mae)
                if child_mae < self.best_mae:
                    self.best_mae = child_mae
                    self.best_rmse = child_rmse
                    self.best_sequence = sequence
                '''if i == SAMPLES_PER_CONTROLLER_EPOCH - 1:
                    if self.window.shift == 1:
                        self.patient_data[current_patient] = [self.best_mae]
                    elif self.window.shift == 6:
                        self.patient_data[current_patient] = [self.best_mae]
                    else:
                        self.patient_data[current_patient].append(self.best_mae)'''

                '''
                plt.plot(forecast_data[5+FORECAST_MINUTES//5:len(predictions_array)], color='black', linestyle='--', label='Actual Value')
                plt.plot(predictions_array, color='blue', label='Predictions')
                plt.legend()
                plt.show()'''
                    
                
                #self.window.plot(model[0])
                for name, model in all_models:
                    preds = model.predict(current_window.test,verbose=0)
                    model_forecasts[name].append(preds)
                    

                '''N = test_df.values.shape[0]
                #print(N)
                print(f"shape of model_fore... is {model_forecasts['current_model'][0].shape}")
                print(f"shape of test_df.values is {test_df.values.shape}")
                #print(f"max of model_fore... is {model_forecasts['current_model'][0].max()}")
                #print(f"max of test_df.values is {test_df.values.max()}")
                mae_ = np.abs( test_df.values[5+int(FORECAST_MINUTES/5):] - np.array(model_forecasts['current_model'][0])).sum() / N
                print('Prediction MAE:',round(mae_,3))
                #print(f"test_df[50:100] is {test_df[50:100]} and model_fore... is {model_forecasts['current_model'][0][50:100]}")'''
                
                

                '''plt.plot(current_window.test_df.values[5+int(FORECAST_MINUTES/5):], color='black', linestyle='--',label='Actual Value')
                plt.plot(model_forecasts['current_model'][0].squeeze(), color='blue',label='Predictions')
                plt.legend()
                plt.show()'''
                                
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')

            xc, yc, mae_target = self.prepare_controller_data(sequences)
            if self.use_predictor:
                self.train_controller(self.controller_model,
                                  xc,
                                  yc,
                                  mae_target[-self.samples_per_controller_epoch:], pred_accuracies)
            else:
                self.train_controller(self.controller_model,
                                  xc,
                                  yc,
                                  mae_target[-self.samples_per_controller_epoch:])
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
        
    
