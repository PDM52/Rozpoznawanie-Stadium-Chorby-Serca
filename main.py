import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging

from model_manager import get_models, select_the_best

toPrint = ''

path = 'heart_disease/'

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal', 'num']
data = pd.read_csv(path + 'processed.cleveland.data', header=None, names=column_names)

data = data.map(lambda x: np.nan if x == '?' else pd.to_numeric(x, errors='coerce'))
data = data.apply(lambda x: x.fillna(x.mean()), axis=0)

if input('Czy zastosowac selekcję istotnych cech? (y/n) ') == 'y':
    X = data.iloc[:, 1:-2]
    y = data.iloc[:, -1]
    model = Lasso(alpha=0.05)
    model.fit(X, y)
    selected_features = X.columns[model.coef_ != 0]
    columns_to_remove = data.columns.difference(selected_features)
    columns_to_remove = columns_to_remove.delete(columns_to_remove.get_loc('num'))
    data = data.drop(columns=columns_to_remove)


columns_to_normalize = data.columns[:-1]
data_to_normalize = data[columns_to_normalize]

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
data[columns_to_normalize] = normalized_data

dataset1 = data.copy()
dataset1['num'] = dataset1['num'].apply(lambda x: 1 if x > 0 else x)

dataset2 = data[data['num'] != 0]

datasets = {'Dane oryginalne':data, 'Chory lub Zdrowy':dataset1, 'Samo stadium choroby':dataset2}

for key in datasets:
    toPrint += '\n'
    toPrint += (key + ':' + '\n')
    training_data, test_data = train_test_split(datasets[key], test_size=0.3, random_state=50, shuffle=True)
    X_train = training_data.iloc[:, 1:-2]
    y_train = training_data.iloc[:, -1]
    X_test = test_data.iloc[:, 1:-2]
    y_test = test_data.iloc[:, -1]

    test_models = get_models()
    for key in test_models:
        best_model = select_the_best(test_models[key][0], test_models[key][1], X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred)*100) + '%'
        toPrint += (key + ": " + accuracy + '\n')

    neuronNetwork = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(10, activation='softmax')
    ])
    neuronNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    neuronNetwork.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    y_pred_prob = neuronNetwork.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred)*100) + '%'
    toPrint += ("Sieć neuronowa: " + accuracy + '\n')

print(toPrint)
