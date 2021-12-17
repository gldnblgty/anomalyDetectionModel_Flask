# Anomaly Detection Dummy LGTM Model - Example

## Local Setup (Windows)

In the following order:

-  Activating Virtual Environment (Windows)

```bash
	pip install virtualenv
	virtualenv venv
	.\venv\Scripts\activate
```

- Installing libs (into venv)
```bash
pip install -r requirements.txt
```

- Running and Saving the model

```bash
(root) python model.py
```

- Consuming the model with a dummy Flask call

```bash
(root) flask run
```

## Important Notes for DS/ML Engineer
- model.summary() isn't supported by keras during export. Kindly be careful to comment it before model exporting. Related error is below:
```bash
 AttributeError: '_UserObject' object has no attribute 'predict'
```

- Based on the model.shape, below is an example anomaly detection based on evaluation metrics. 
This only works at the same time with training, a DS/ML engineer should vectorize the datasets 
based on input type (csv, stream, single value...etc.) and deployment environment (Docker, API, Databricks ..etc.)

```bash
# Below is an example for the developer/ML engineer to consume/containerize the model. 
# Definition of anomaly from evalution metrics perspective
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

X_test_pred = model.predict(X_test, verbose=0)
threshold = np.max(train_mae_loss)

test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

test_score_df = pd.DataFrame(test[TIME_STEPS:])
#print(f'test_score_df : {test_score_df}')
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['value'] = test[TIME_STEPS:]['value']

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
print(f'anomalies : {anomalies}')

lm.load_model(X_test)
```