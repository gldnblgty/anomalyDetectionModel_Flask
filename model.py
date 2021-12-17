from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

import pandas as pd
import numpy as np
import utils as utils

# Dataset:https://www.kaggle.com/boltzmannbrain/nab
df = pd.read_csv('art_daily_small_noise.csv')
df = df[['timestamp', 'value']]
df['timestamp'] = pd.to_datetime(df['timestamp'])
TIME_STEPS=30
#print(df)

train, test = df.loc[df['timestamp'] <= '2014-04-06'], df.loc[df['timestamp'] > '2014-04-07']

X_train, y_train = utils.split_sequences(train[['value']], train['value'])
X_test, y_test = utils.split_sequences(test[['value']], test['value'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')

#model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)

model.save('saved_model')
print(f'model saved')

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
#print(f'train_mae_loss : {train_mae_loss}')

X_test_pred = model.predict(X_test, verbose=0)
threshold = np.max(train_mae_loss)
#print(f'threshold : {threshold}')

test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)
#print(f'test_mae_loss : {test_mae_loss}')

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['value'] = test[TIME_STEPS:]['value']

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
print(f'anomalies : {anomalies}')