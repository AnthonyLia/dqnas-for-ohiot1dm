import CNNGenerator
import tensorflow
import tensorflow.keras
import numpy as np
from time_series2 import *
import matplotlib.pyplot as plt
import gc
from CNNCONSTANTS import *

def compile_model(model, lr, decay, epsilon):
    optim=tensorflow.keras.optimizers.legacy.Adam( learning_rate = lr, decay = decay )
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), optimizer=tensorflow.keras.optimizers.Adam(epsilon=epsilon), metrics=tensorflow.keras.metrics.MeanAbsoluteError())

def evaluate(model, window):
    return model.evaluate(window.test, verbose=1)


results = []
patient_numbers = [540,544,552,559,563,567,570,575,584,588,591,596]
patient = {540:np.array([1936,18725,16931,64148,64150]),
           544:np.array([4088,10268,17915,64148,64150]),
           552:np.array([7641,989,4784,64148,64150]),
           559:np.array([602,19892,45952,64148,64150]),
           563:np.array([1316,20240,16337,64148,64150]),
           567:np.array([3896,9119,33563,64148,64150]),
           570:np.array([11001,867,48357,64148,64150]),
           575:np.array([4815,19634,20726,64148,64150]),
           584:np.array([3894,18635,22664,64148,64150]),
           588:np.array([3657,27772,8425,64148,64150]),
           591:np.array([2888,23274,9714,64148,64150]),
           596:np.array([4282,1346,4228,64148,64150]) }

for patient_num in patient:
    best_mae = 999
    for lr in [0.05, 0.07, 0.09, 0.105, 0.12, 0.135]:
        for batch in [8,16]:
            for epsilon in [10e-7, 10e-6, 10e-5]:
                if lr == 0.05 and batch == 8 and epsilon == 10e-7:
                    print(f'\nstarting patient {patient_num}')
                w1 = WindowGenerator(patient_num, input_width=6, label_width=1, shift=1,
                                     label_columns=['Glucose Level'])
                search_space = CNNGenerator.CNNSearchSpace(1)
                vocab = search_space.vocab
                generator = CNNGenerator.CNNGenerator()
                model = generator.create_model(patient[patient_num],(6,1))
                compile_model(model,lr,0.0,epsilon)
                history = model.fit(w1.train, epochs=370, batch_size=batch, validation_data=w1.val,callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=30) ],verbose=0)
                result = model.evaluate(w1.test, verbose=0)
                if result[1] < best_mae:
                    best_mae = result[1]
                    print(f'saving patient {patient_num} weights with mae {best_mae}')
                    model.save(f'{patient_num}_weights.keras')

'''
for lr in [0.001, 0.003, 0.006, 0.01, 0.03, 0.05, 0.07, 0.09, 0.105, 0.12, 0.135, 0.15, 0.165]:
    for batch in [8,16,32]:
        for decay in [0.0]:
            for epsilon in [10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3]:
                results.append(f'lr:{lr}, epsilon:{epsilon}, batch_size:{batch}')
                model = generator.create_model(example_sequence4, (6,1))
                compile_model(model,lr,decay,epsilon)
                history = model.fit(w2.train,
                            epochs=370,
                            batch_size=batch,
                            validation_data=w2.val,
                            callbacks=[ tensorflow.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=30) ],
                            verbose=1)
                results.append(history.history['mean_absolute_error'][-5:])
                results.append(evaluate(model,w2)[1])
'''


fig, ax=plt.subplots()
img = plt.imread('seg_canvas.jpg')
ax.imshow(img,extent=[0,600,0,600])

forecast_data = w1.test_df[:-1].values
predictions_array = np.zeros((len(w1.test_df.values)-FORECAST_MINUTES//5-6,1))
for i in range(6,len(w1.test_df.values) - FORECAST_MINUTES//5-6):
    current_window = forecast_data[i:i+6]
    #for j in range(FORECAST_MINUTES//5-1):
    predict_window = current_window #np.append(current_window[j if j<6 else 6:],pseudo_ground_truth[:j])[None,:,None]
                        #print(predict_window)
                        #print(predict_window.shape)
                        
    #    pseudo_ground_truth[j] = model[0].predict(predict_window,verbose=0)
                        
    #predict_window = np.append(current_window[j+1 if j+1<6 else 6:],pseudo_ground_truth[:j+1])[None,:,None]
    predictions_array[i] = model.predict(predict_window[None,:],verbose=0)
    #if i % 50 == 0:
     #   del pseudo_ground_truth
      #  del current_window
       # del predict_window
        #gc.collect()

w1.test_df *= w1.train_std.iloc[0]
w1.test_df += w1.train_mean.iloc[0]

predictions_array *= w1.train_std.iloc[0]
predictions_array += w1.train_mean.iloc[0]

for i in range(min([len(predictions_array),len(w1.test_df)])):
    plt.scatter(w1.test_df.values[i],predictions_array[i], color='blue',s=1)
                    
