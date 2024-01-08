import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import autoviz
#from statsmodels.tsa.seasonal import seasonal_decompose
#Scikit learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Tensorflow/Keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

# import data
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/self_learning/sudeste_reduced.csv', index_col = 'mdct', parse_dates = True)
df = pd.DataFrame(data, columns = ['wdsp'])
df = data['2012':'2014']
wdsp = df['wdsp'] # wind speed

# look for missing data, if greater than 0 then correct in following steps
if (df==0).sum(axis=0) > 0:
    imputer = SimpleImputer(missing_values=0, strategy='mean') # it's better to replace data with the mean values than
    # erasing it. Timeseries models need the previous values for prediction so missing values are memory loss
    imputer.fit(df)
    imputed_data = imputer.fit_transform(df)
    full_data = df
    full_data['wdsp'] = imputed_data
else:
  full_data = df

# visualize data
def time_series_multiline(df, timelike_colname, value_colname, series_colname, figscale=1.2, mpl_palette_name='Dark2'):

  figsize = (10 * figscale, 5.2 * figscale)
  palette = list(sns.palettes.mpl_palette(mpl_palette_name))
  def _plot_series(series, series_name, series_index=0):
    if value_colname == 'count()':
      counted = (series[timelike_colname]
                 .value_counts()
                 .reset_index(name='counts')
                 .rename({'index': timelike_colname}, axis=1)
                 .sort_values(timelike_colname, ascending=True))
      xs = counted[timelike_colname]
      ys = counted['counts']
    else:
      xs = series[timelike_colname]
      ys = series[value_colname]
    plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

  fig, ax = plt.subplots(figsize=figsize, layout='constrained')
  df = df.sort_values(timelike_colname, ascending=True)
  if series_colname:
    for i, (series_name, series) in enumerate(df.groupby(series_colname)):
      _plot_series(series, series_name, i)
    fig.legend(title=series_colname, bbox_to_anchor=(1, 1), loc='upper left')
  else:
    _plot_series(df, '')
  sns.despine(fig=fig, ax=ax)
  plt.xlabel(timelike_colname)
  plt.ylabel(value_colname)
  return autoviz.MplChart.from_current_mpl_state()

chart1 = time_series_multiline(df, *['mdct', 'wdsp', None], **{})

def histogram(df, colname, num_bins=20, figscale=0.5):
  df[colname].plot(kind='hist', bins=num_bins, title=colname, figsize=(8*figscale, 4*figscale))
  plt.gca().spines[['top', 'right',]].set_visible(False) # this erases the half of the frame
  plt.tight_layout()
  return autoviz.MplChart.from_current_mpl_state()

chart2 = histogram(df, *['wdsp'], **{})

# separate into train/test
train_size = round(0.8*len(full_data))
print(f'this is the training size {train_size}')
test_size = round(0.2*len(full_data))
X_train = full_data[:train_size]
print(f'this is the training data {X_train}')
X_test = full_data[train_size:]

# scale based on train data otherwise information leak from test to train
scaler = StandardScaler()
X_train_values = scaler.fit_transform(X_train)
X_test_values = scaler.transform(X_test)
X_train['wdsp'] = X_train_values
X_test['wdsp'] = X_test_values

#define generator
n_input = 3000
n_features = 1
# This class generates batches of input-output pairs from time-series data, it takes 
# sequential data as input and creates sliding windows of data along with corresponding TARGET values.
generator = TimeseriesGenerator(X_train, X_train, length = n_input, batch_size = 1) # batch is similar to k-fold

# Analize generator output
X0,y0 = generator[0] # just analyze one of the two outputs, nevertheless both will be used later for training
print(f'Given the Array: \n{X0.flatten()}') # flatten() returns a copy of the array collapsed into one dimension.
print(f'Predict this y: \n {y0}')
# X,y are the normalized time, wdsp

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# fit and save the model
model.fit(generator, epochs = 1)
model.save('/content/drive/MyDrive/Colab Notebooks/lstm_model.keras')