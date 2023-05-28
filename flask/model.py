import pickle
#.\venv\Scripts\activate
from yahoo_fin.stock_info import get_data
from datetime import datetime

#Getting data for the last 4 days.
import datetime

now = datetime.datetime.now()
formatted_date = now.strftime("%m/%d/%Y")

# Add one day
one_day = datetime.timedelta(days=1)
five_day = datetime.timedelta(days=5)

new_date = datetime.datetime.strptime(formatted_date, "%m/%d/%Y") + one_day 
old_date = datetime.datetime.strptime(formatted_date, "%m/%d/%Y") - five_day

formatted_new_date = new_date.strftime("%m/%d/%Y")
formatted_old_date=old_date.strftime("%m/%d/%Y")

formatted_new_date

#data= get_data("ADANIPORTS.NS", start_date="03/18/2023", end_date=formatted_new_date, index_as_date = True, interval="1m")
data= get_data("TATASTEEL.NS", start_date=formatted_old_date, end_date=formatted_new_date, index_as_date = True, interval="1m")
data

#Removing timestamp as index.
data.reset_index(inplace=True)
data.rename(columns={'index': 'timestamp'}, inplace=True)
data

#Coverting UTC time to IST.
import pandas as pd
# localize the timestamp column to the UTC timezone
data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize('UTC')
# convert the timestamp column to the IST timezone with the desired time zone offset
data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
# format the timestamp column as a string in the desired format
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
data

data.tail(20)

#Converting string or numeric format to a datetime format using the pd.to_datetime
import pandas as pd

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data

#Checking null values
data.isnull().sum()

#Dropping null values
data=data.dropna()
data

data.isnull().sum()

time_steps = 60

#1st 60 timesteps are used to give result for 61st timestep
data1=data.iloc[time_steps+1:,:]

data1.shape
data1

mini=data1['close'].min()
mini

maxi=data1['close'].max()
maxi

#Normalisation of data
from sklearn.preprocessing import MinMaxScaler

# select only numeric columns for normalization
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
data_numeric = data[numeric_cols]

# normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_numeric)

# replace the original values with the normalized values
data[numeric_cols] = data_normalized
data

data.to_csv('datanormalised.csv', index=False)

data['close'].isnull().sum()


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

numeric_cols = ['open', 'high', 'low', 'close', 'volume']
data_numeric = data[numeric_cols]

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_numeric)

# Define the number of time steps
time_steps = 60

# Create input sequence
X = []
y = []
for i in range(time_steps, len(data_scaled)):
    X.append(data_scaled[i-time_steps:i, :])
    y.append(data_scaled[i, [0, 1, 2, 3, 4]])  # use columns 0, 1, 4, and 3 for open, high, volume, and close, respectively
X, y = np.array(X), np.array(y)

print(X)

y

#Generating output file for X
import numpy as np
import csv

# Create a 3D array
arr_3d = X

# Reshape the 3D array into a 2D array
arr_2d = arr_3d.reshape(-1, arr_3d.shape[-1])

# Write the 2D array to a CSV file
with open('X3d.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(arr_2d)

#Generating output file for y
with open('y.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each element of the array as a separate row
    for element in y:
        writer.writerow([element])

X.shape

y.shape

# Split data into training and testing sets
split_ratio = 0.7
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM

#model = Sequential()
#model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X.shape[2])))
#model.add(LSTM(units=50, return_sequences=True))
#model.add(LSTM(units=50))
#model.add(Dense(units=y.shape[1]))  # change output dimension to match the number of columns in y
#model.compile(optimizer='adam', loss='mean_squared_error')

model = Sequential()
model.add(LSTM(units=25, return_sequences=True, input_shape=(time_steps, X.shape[2])))
model.add(LSTM(units=25, return_sequences=True))
model.add(LSTM(units=25))
model.add(Dense(units=y.shape[1]))  # change output dimension to match the number of columns in y
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
#model.fit(X_train, y_train, epochs=50, batch_size=32)
model.fit(X_train, y_train, epochs=100, batch_size=32)

#Hyperparameter Tuning
#!pip install scikeras
#from scikeras.wrappers import KerasRegressor
#from sklearn.model_selection import GridSearchCV

#def create_model(units=50):
#    model = Sequential()
#    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps, X.shape[2])))
#    model.add(LSTM(units=units, return_sequences=True))
#    model.add(LSTM(units=units))
#    model.add(Dense(units=y.shape[1]))
#    model.compile(optimizer='adam', loss='mean_squared_error')
#    return model

#model = KerasRegressor(build_fn=create_model, verbose=0, units=25)

# Define hyperparameters to search over
#param_grid = {
#    'units': [25, 50, 75],
#    'epochs': [50, 100],
#    'batch_size': [32, 64]
#}

# Perform grid search
#grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
#grid_result = grid.fit(X_train, y_train)

# Print results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

X_train

X_test

y_pred = model.predict(X_test)
y_pred.shape

y_pred

#Evaluating rmse value for model
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse

#Generating output files
with open('ypred.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each element of the array as a separate row
    for element in y_pred:
        writer.writerow([element])

with open('ytest.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write each element of the array as a separate row
    for element in y_test:
        writer.writerow([element])

y_pred.shape

y_test.shape

data1.shape

j=int(len(data1))
j

p=int(0.7*len(data1))
p

i=int(len(data1)-0.7*len(data1))
i

sz=j-p

import pandas as pd

if (y_test.shape[0]>sz):
  data2 = data1.iloc[sz-1:]
else:
  data2 = data1.iloc[-sz:]
# print the data1_last_30pct dataframe
data2

# Add the 1D arrays as new columns to the DataFrame
#data2['YTest'] = 
print(data2.shape(),y_test)
data2.loc[:, 'YTest'] = y_test[:,3]
data2.loc[:, 'YPred'] = y_pred[:,3]

# Print the DataFrame to verify the new columns have been added
data2

# Denormalize the column 'y_pred'
data2.loc[:, 'YPred'] = data2['YPred'] * (maxi - mini) + mini

data2.loc[:, 'YTest'] = data2['YTest'] * (maxi - mini) + mini
data2

data2 = data2.reset_index()
data2

data2.to_csv('data2.csv', index=False)

# create a DataFrame with actual and predicted values
df = data2[['YTest', 'YPred']]

# plot the data using matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data2['timestamp'], df['YTest'], color='blue', label='Actual')
ax.plot(data2['timestamp'], df['YPred'], color='red', label='Predicted')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Closing Price')
ax.legend()

# rotate the x-tick labels by 45 degrees
plt.xticks(rotation=45)

plt.show()

# Prepare the input data for prediction
# Assume that X_test contains the last 60 minutes of the original data
import numpy as np
last_60_minutes = X_test[-1]  # shape: (60, number of features)
prediction_input = np.reshape(last_60_minutes, (1, time_steps, X.shape[2]))  # shape: (1, 60, number of features)

# Make the prediction
predicted_values = model.predict(prediction_input)  # shape: (1, y.shape[1])

# Initialize the predicted_values_array
predicted_values_array = predicted_values.copy()

# Repeat the process for the next 59 minutes
for i in range(59):
    # Shift the last_60_minutes array by one minute
    last_60_minutes[:-1] = last_60_minutes[1:]
    
    # Set the last row of last_60_minutes to the predicted values from the previous minute
    last_60_minutes[-1] = predicted_values[0]
    
    # Prepare the input data for prediction
    prediction_input = np.reshape(last_60_minutes, (1, time_steps, X.shape[2]))
    
    # Make the prediction for the next minute
    predicted_values = model.predict(prediction_input)
    
    # Append the predicted values to the output array
    predicted_values_array = np.append(predicted_values_array, predicted_values, axis=0)

print(predicted_values_array)

predicted_values_array.shape

predicted_values_array[:,3]

predicted_values_array[:,3]=predicted_values_array[:,3] * (maxi - mini) + mini
predicted_values_array[:,3]

import matplotlib.pyplot as plt
plt.plot(predicted_values_array[:,3])
plt.xlabel('Minutes')
plt.ylabel('Stock Price')
plt.title('Predicted Stock Price for the Next 60 Minutes')
plt.show()

pickle.dump(model,open('model.pkl','wb'))

lmodel=pickle.load(open('model.pkl','rb'))