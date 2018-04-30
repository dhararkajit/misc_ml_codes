from __future__ import print_function

import os
import numpy as np
import pandas as pd
import random
import math
import sklearn.datasets as ds
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


if __name__ == "__main__":
    # Load Boston Data
    boston = ds.load_boston()
    X = boston.data
    y = boston.target/50.
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33,random_state=42)


    nor = StandardScaler()
    X_train = nor.fit_transform(X_train)
    X_test = nor.transform(X_test)    


    model =MLPRegressor(activation='logistic', alpha=0.01, max_iter=1000, hidden_layer_sizes=(200,),learning_rate_init=0.1,solver='lbfgs',learning_rate='adaptive',momentum=0.1,tol=0.01)  # replace with your own ML model here
    model.fit(X_train,y_train)

    _CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    _SERIALIZATION_DIR = os.path.join(_CUR_DIR, "..", "models", "boston")

    if not os.path.exists(_SERIALIZATION_DIR):
        os.makedirs(_SERIALIZATION_DIR)
    model_filename = os.path.join(_SERIALIZATION_DIR, "model.pkl")

    joblib.dump(model, model_filename)
    print("Successfully Built and Picked into models folder")
