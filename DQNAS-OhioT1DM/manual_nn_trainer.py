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

example_sequence1 = np.array([17264, 28171, 55537, 64148, 64150])
example_sequence2 = np.array([6641, 23379, 36085, 64148, 64150])
example_sequence3 = np.array([305, 7278, 62795, 64148, 64150]) #0.2101847976
example_sequence4 = np.array([16372, 9782, 52632, 64148, 64150]) #0.210743815

five_min_sequence_570_1 = np.array([12758, 4479, 28650, 64148, 64150]) #0.039715927
five_min_sequence_570_2 = np.array([9802, 62080, 28389, 64148, 64150]) #0.039848685
five_min_sequence_584_1 = np.array([20139, 33152, 736, 64148, 64150]) #0.05833883956

patient = 584
w1 = WindowGenerator(patient, input_width=6, label_width=1, shift=1,
                     label_columns=['Glucose Level'])
w2 = WindowGenerator(patient, input_width=6, label_width=1, shift=6,
                     label_columns=['Glucose Level'])

results = []

search_space = CNNGenerator.CNNSearchSpace(1)
vocab = search_space.vocab
generator = CNNGenerator.CNNGenerator()

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



model = generator.create_model(five_min_sequence_584_1,(6,1))
compile_model(model,0.1,0.0,10e-6)
history = model.fit(w1.train, epochs=370, batch_size=16, validation_data=w1.val,callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=30) ],verbose=0)
print('gotpast history')

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
                    
