
# ## Below part is a test to check if model is working. 
# ## This should also be transferred to Flask API and rest is same as starting point of line 51
# loaded_model = keras.models.load_model("saved_model")
# print(f'model reloaded')

# result = loaded_model.predict(X_test)
# print(f'prediction : {result}')



# # From Gulden
# # TO DO Katya; below is evalution metrics based on my data tests 
# # or another say to say is : definition of anomaly - this is for Flask call during containerization
# train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
# #print(f'train_mae_loss : {train_mae_loss}')

# X_test_pred = model.predict(X_test, verbose=0)
# threshold = np.max(train_mae_loss)
# #print(f'threshold : {threshold}')

# test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)
# #print(f'test_mae_loss : {test_mae_loss}')

# test_score_df = pd.DataFrame(test[TIME_STEPS:])
# #print(f'test_score_df : {test_score_df}')
# test_score_df['loss'] = test_mae_loss
# test_score_df['threshold'] = threshold
# test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
# test_score_df['value'] = test[TIME_STEPS:]['value']

# anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
# print(f'anomalies : {anomalies}')