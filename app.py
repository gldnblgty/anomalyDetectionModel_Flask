from flask import Flask
from flask import request
import json
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def call_prediction():
    loaded_model = tf.keras.models.load_model("saved_model")
    
    #static_input - feel free to adapt dynamically in your scenario
    input = 50;

    #print(X_regression_set.shape) - #(200, 1)
    X_regression_set = np.expand_dims(np.arange(0, 100, 2), axis=1 )
    X_regression_set_resized = np.expand_dims(X_regression_set, axis=1 ) 
    
    #print(X_regression_pred.shape) - #(200, 30, 1)
    X_regression_pred = loaded_model.predict(X_regression_set, verbose=0)
    test_mae_loss = np.mean(np.abs(X_regression_pred - X_regression_set_resized), axis=1)
    threshold = np.max(test_mae_loss)

    try:
        if input > threshold:
            output_str = "Anomaly Detected"
        else:
            output_str = "No Anomaly Detected"
    except ValueError:
        output_str = "Something unexpected.. probably threshold couldn't be obtained by the model because of broken values."
    
    return output_str

if __name__=='__main__':
   app.run()