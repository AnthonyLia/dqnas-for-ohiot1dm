# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:40:18 2022

@original_author: AnshumaanChauhan

"""
'''
Edited by Anthony Liardo
'''

import CNNCONSTANTS
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from NASutils import *
from cnnnas import CNNNAS
from CNNCONSTANTS import TOP_N
from tensorflow.keras.utils import to_categorical
#from time_series2 import *

'''
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
train_y=to_categorical(train_y,num_classes=10)
train_new_X=train_X[:100]
train_new_y=train_y[:100]'''



patient_numbers = [540,544,552,559,563,567,570,575,584,588,591,596]
#patient_numbers = [563]
patient_data = {540:{},544:{},552:{},559:{},563:{},567:{},570:{},575:{},584:{},588:{},591:{},596:{}}

for patient in patient_numbers:
    PATIENT_NUMBER = patient
    from time_series2 import *
    w1 = WindowGenerator(PATIENT_NUMBER, input_width=6, label_width=1, shift=1,
                     label_columns=['Glucose Level'])
    w2 = WindowGenerator(PATIENT_NUMBER, input_width=6, label_width=1, shift=6,
                     label_columns=['Glucose Level'])
    w3 = WindowGenerator(PATIENT_NUMBER, input_width=6, label_width=1, shift=9,
                     label_columns=['Glucose Level'])
    patient_data[patient]['standard_deviation'] = w1.train_std
    for window in [w1]:
        nas_object = CNNNAS(window)
        #print(f'train_std is {window.train_std}')
        #print(f'patientnumber is {PATIENT_NUMBER}')
        print(f'patient is {patient}')
        data = nas_object.search(patient,window)
        patient_data[patient][window] = [nas_object.best_sequence,nas_object.best_mae*window.train_std,nas_object.best_rmse*window.train_std]
        with open('test.txt','w') as file:
            file.write(str(patient_data))
        


