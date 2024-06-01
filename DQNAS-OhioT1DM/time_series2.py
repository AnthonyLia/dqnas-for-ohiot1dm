'''
Created by: Anthony Liardo
Based on TensorFlow time-series tutorial
'''
import datetime
import IPython
import IPython.display
import numpy as np
import pandas as pd
import tensorflow as tf
import xml_reader
from CNNCONSTANTS import FORECAST_MINUTES
import matplotlib.pyplot as plt
#from NASrun import PATIENT_NUMBER



def make_df(patient_number):
  train_df = pd.DataFrame(xml_reader.get_glucose_train_data(patient_number), columns=['Date Time','Glucose Level'])
  train_date_time = pd.to_datetime(train_df.pop('Date Time'), format='%d-%m-%Y %H:%M:%S')
  timestamp_s = train_date_time.map(pd.Timestamp.timestamp)
  train_column_indices = {name: i for i, name in enumerate(train_df.columns)}

  test_df = pd.DataFrame(xml_reader.get_glucose_test_data(patient_number), columns=['Date Time','Glucose Level'])
  test_date_time = pd.to_datetime(test_df.pop('Date Time'), format='%d-%m-%Y %H:%M:%S')
  timestamp_s = test_date_time.map(pd.Timestamp.timestamp)
  test_column_indices = {name: i for i, name in enumerate(test_df.columns)}

  n = len(train_df)
  val_df = train_df[int(n*0.9):]
  train_df = train_df[0:int(n*0.9)]

  num_features = train_df.shape[1]

  train_mean = train_df.mean()
  train_std = train_df.std()
  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std
  test_df = (test_df - train_mean) / train_std

  return train_df,val_df,test_df,train_mean,train_std

'''train_set = tf.data.Dataset.from_tensor_slices(train_df.values)
train_set = train_set.window(6+int(FORECAST_MINUTES/5), shift=1, drop_remainder=True)
train_set = train_set.flat_map(lambda x: x.batch(6+int(FORECAST_MINUTES/5)))
train_set = train_set.map(lambda x: (x[:-1], x[-1]))
#train_set = train_set.shuffle(1_000)
train_set = train_set.batch(32).prefetch(1)
#train_set = train_set.batch(1)

val_set = tf.data.Dataset.from_tensor_slices(val_df.values)
val_set = val_set.window(6+int(FORECAST_MINUTES/5), shift=1, drop_remainder=True)
val_set = val_set.flat_map(lambda x: x.batch(6+int(FORECAST_MINUTES/5)))
val_set = val_set.map(lambda x: (x[:-1], x[-1]))
val_set = val_set.batch(32).prefetch(1)
#val_set = val_set.batch(1)

test_set = tf.data.Dataset.from_tensor_slices(test_df.values)
test_set = test_set.window(6+int(FORECAST_MINUTES/5), shift=1, drop_remainder=True)
test_set = test_set.flat_map(lambda x: x.batch(6+int(FORECAST_MINUTES/5)))
test_set = test_set.map(lambda x: (x[:-1], x[-1]))
test_set = test_set.batch(32).prefetch(1)
#test_set = test_set.batch(1)
'''

class WindowGenerator():
  def __init__(self, patient_number, input_width, label_width, shift,
               #train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df,self.val_df,self.test_df,self.train_mean,self.train_std = make_df(patient_number)
    #self.train_df = train_df
    #self.val_df = val_df
    #self.test_df = test_df
    self.patient_number = patient_number
    
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(self.train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window



class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]



def make_dataset(self, data, batch_size):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=batch_size,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df, int(len(self.train_df)/self.total_window_size))
  #return self.make_dataset(self.train_df, 32)
@property
def val(self):
  return self.make_dataset(self.val_df, int(len(self.val_df)/self.total_window_size))
  #return self.make_dataset(self.val_df, 32)
@property
def test(self):
  return self.make_dataset(self.test_df, int(len(self.test_df)/self.total_window_size))
  #return self.make_dataset(self.test_df, 32)
@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example





def plot(self, model=None, plot_col='Glucose Level', max_subplots=1):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue
    #print(labels)
    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [5m]')

WindowGenerator.plot = plot





val_performance = {}
performance = {}
MAX_EPOCHS = 20




def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

'''
def compile_and_fit2(model, train_set, val_set, patience=3):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                #optimizer=tf.keras.optimizers.legacy.Adam(lr=0.1),
                optimizer=tf.keras.optimizers.Adam(epsilon=1e-5),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(train_set, epochs=MAX_EPOCHS,
                      validation_data=val_set,
                      callbacks=[early_stopping],
                      verbose=0)
  return history'''

CONV_WIDTH = 3




conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,)),
                           #activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])



lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

cnn_model = tf.keras.models.Sequential([
    # add extra axis to input data
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[7]),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1,
            activation='relu',
           padding='causal', ),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1,
            activation='relu',
           padding='causal', ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1)
])

test_model = tf.keras.models.Sequential([
  tf.keras.layers.DepthwiseConv1D(kernel_size=1, strides=11,input_shape=np.array([6,1]),padding='same'),
  tf.keras.layers.Conv1DTranspose(filters=224,strides=2,kernel_size=2,padding='same'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1,activation='linear')

]  )


#------------------------------------------------------------
# Multi-Step
#------------------------------------------------------------
multi_val_performance = {}
multi_performance = {}
OUT_STEPS = 24
'''multi_window = WindowGenerator(540,input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)
num_features = multi_window.train_df.shape[1]

multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
'''
'''
history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)
'''
