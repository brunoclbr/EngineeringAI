import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# Scikit learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Tensorflow/Keras
from tensorflow import keras
#from keras.preprocessing.sequence import TimeseriesGenerator ##deprecated
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model

'''
Always keep in mind:
- Confirm periodicity of data, otherwise no timeseries (min requirement for forecasting)
- When working with TS, is important to use val and test data that is more recent (you're trying to predict the future given the past)
- A RNN is a for loop that reuses quantities computed during the previous interation of the loop (keep track of memory)
- Use of bidirectional networks can be useful for text processing, not for forcasting
- FF or CN produce information loss (no memory)
- LSTM fights the vanishing-gradient problem (meaning the earlier training data) as they keep track of "old memory" 
'''

# import and preprocess data. Hourly wheather data in the city of Vitoria, Brazil
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/self_learning/sudeste_reduced.csv', index_col = 'mdct', parse_dates = True)
print(data.columns)
data_wdsp = pd.DataFrame(data, columns = ['wdsp'])
data_wdsp_slice = data_wdsp['2012':'2014']
#decide which columns are not necesary and erase them:
cols = ['wsid','wsnm','stp','elvt','lat', 'lon', 'inme', 'city', 'prov', 'date',
       'yr', 'mo', 'da', 'hr','prcp', 'stp', 'smax', 'smin','gbrd', 'tmin', 'tmax', 'dmin','dmax', 'dewp', 'hmax', 'hmin', 'wdct', 'gust']
data_all_slice = data.drop(cols, axis=1, inplace=False)
data_all_slice = data_all_slice['2012':'2014']
print(data_all_slice)

# Visualize data
def distribution(df):
  plt.figure(figsize = (9,4))
  plt.title("Wind speed distribution between 2007 and 2017")
  plt.ylabel("Wind speed in m/s")
  plt.plot(df)
  plt.show()

def histogram(df, colname, num_bins=15, figsize=(9,4)):
  return df[colname].plot(kind='hist', bins=num_bins, title="Wind speed in m/s", figsize=figsize)

wdsp_distribution = distribution(data_wdsp)
wdsp_training_frequency = histogram(data_wdsp_slice, *['wdsp'], **{})

#results = seasonal_decompose(data_wdsp_slice['wdsp'])
#results.plot()

# Data preprocessing

# impute missing values
def data_imputer(data_frame):
  for (columnName, columnData) in data_frame.iteritems():
    number_of_zeros = (columnData.values==0).sum(axis=0)
    # look for missing data, if greater than 0 then correct in following steps
    if number_of_zeros > 0:
        twoD_data = columnData.values.reshape(-1, 1)      
        imputer = SimpleImputer(missing_values=0, strategy='mean') # it's better to replace data with the mean values than
        # erasing it. Timeseries models need the previous values for prediction so missing values are memory loss
        imputer.fit(twoD_data)
        data_frame[columnName] = imputer.fit_transform(twoD_data)
    else:
      print('data column has no missing values')
  return data_frame

data_wdsp_slice = data_imputer(data_wdsp_slice)
full_data = data_imputer(data_all_slice)

print(full_data)
print((full_data==0).sum(axis=0))
print(full_data.size)

# separate data into test/val/train
print(type(full_data))
train_size = round(0.75*len(full_data))
val_size = round(0.2*len(full_data))
test_size = round(0.05*len(full_data))
idx_val_end = train_size + val_size
idx_test_end = idx_val_end + test_size
# it's better to pass these indexes in the generator so that the targets also get separated.
# work around: double-slice the "targets" in the generator
X_train = full_data.iloc[:train_size]
X_val = full_data.iloc[train_size:idx_val_end]
X_test = full_data.iloc[train_size+val_size:idx_test_end]
print(f'this is the training size: {train_size}')
print(f'this is the training data: \n {X_train}')
print(type(X_train))

# trying different scaling methods

# Warning: A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead. This is usually problematic
# if you're chaining indexing with dataframes, e.g., df['col2:col4]['row2'] = 100

# normalize with mean and std on my own
mean = X_train['wdsp'].mean(axis=0)
std = X_train['wdsp'].std(axis=0)
x_min_mean = X_train['wdsp'] - mean
x_norm = x_min_mean/std
X_train.loc[:,'wdsp'] = x_norm
print(X_train)
# normalize with mean and std with scikit learn and compare with previous normalization
scaler = StandardScaler()
scaled_train_std = scaler.fit_transform(X_train)
scaled_val_std = scaler.transform(X_val)
scaled_test_std = scaler.transform(X_test)
print(f'Scaling with mean and std: \n{scaled_train_std}\n')
# normalize with max min values with scikit learn
scaler2 = MinMaxScaler()
scaler2.fit(X_train)
scaled_train_minmax = scaler2.transform(X_train)
scaled_val_minmax = scaler.transform(X_val)
scaled_test_minmax = scaler2.transform(X_test)
print(f'Scaling with max and min values: \n {scaled_train_minmax}')

# TimeSeries generator

# define generators
'''
The generator class generates batches of input-output pairs from time-series data.
It takes sequential data as input and creates sliding windows of data along with
corresponding TARGET values. Since the sample N and N+1 are highly redundant, it would
be wasteful to allocate memory for every single timestamp. Instead, we generate samples
on the fly while only keeping in memory the full_data and wdsp.
'''
sequence_lenght = 120 # window length of generator, observations will go back 5 days (120 hours)
# later input size of model = on sewuence window as a rank 1 tensor. Commonly defined as (sequence_length, full_data.shape[-1])
# don't confuse pd.dataframe.shape with np.reshape
model_input_size = (sequence_lenght,full_data.shape[-1])
delay = sequence_lenght + 24 - 1 # the target of a sequence will be the wdsp 24 hours after the end of the sequence

generator_train = keras.utils.timeseries_dataset_from_array(data=scaled_train_std[:-delay],
                                                            targets=data_wdsp_slice[:train_size][delay:],
                                                            sequence_length = sequence_lenght,
                                                            shuffle=True,
                                                            sampling_rate=1,
                                                            batch_size=2)
# list = [0,1,2,3,4,5,6,7,8,9] and delay = 3 then data[:-3]= [0,1,2,3,4,5,6]
# and targets[3:] = [3,4,5,6,7,8,9]. Last sequence would be [4,5,6], 7.

generator_val = keras.utils.timeseries_dataset_from_array(data=scaled_val_std[:-delay],
                                                            targets=data_wdsp_slice[train_size:idx_val_end][delay:],
                                                            sequence_length = sequence_lenght,
                                                            shuffle=True,
                                                            sampling_rate=1,
                                                            batch_size=2)

generator_test = keras.utils.timeseries_dataset_from_array(data=scaled_test_std[:-delay],
                                                            targets=data_wdsp_slice[train_size+val_size:idx_test_end][delay:],
                                                            sequence_length = sequence_lenght,
                                                            shuffle=True,
                                                            sampling_rate=1,
                                                            batch_size=2)

# Analize generator output
for inputs, targets in generator_train:
  print(inputs)
  print(targets)
  for i in range(inputs.shape[0]):
    print([x for x in inputs[i]], targets[i])
  break
print(f' This is the length of my generator (#batches): \n {len(generator_train)}')

# Create LSTM model

# define functional model
inputs = keras.Input(shape=model_input_size)
x = LSTM(32, recurrent_dropout=0.25)(inputs)
x = Dropout(0.5)(x)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
#callbacks
callbacks = [keras.callbacks.ModelCheckpoint("wdsp_lstm.keras",
                                             save_best_only=True)]
#compile
model.compile(optimizer="adam", loss="mse", metrics="mae")
model.summary() #The None in the output shape of a layer means that the model can handle input sequences of varying batch sizes.

# or define sequential model
### model2 = Sequential()
### model2.add(LSTM(100, activation='relu', input_shape=(120, 1)))
### model2.add(Dense(1))
### model2.compile(optimizer='adam', loss='mse')
### model2.summary()
### model2.fit(generator_train, epochs = 1)

#fit
history = model.fit(generator_train,
                    epochs=10,
                    validation_data=generator_val,
                    callbacks=callbacks)

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save, load, predict

model.save('/content/drive/MyDrive/Colab Notebooks/lstm_model.keras')
loaded_model = load_model('/content/drive/MyDrive/Colab Notebooks/lstm_model.keras')
wdsp_pred = model.predict(generator_test)