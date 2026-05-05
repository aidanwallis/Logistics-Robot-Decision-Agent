# CPSC 481 - Logistics Robot Decision Agent ml.py
# File Author: Nicholas Reardon

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def load_and_train_model():
    data = pd.read_csv('logistics_robot_data.csv')
    
    encoders = {}
    for column in data.columns:
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column]) # type: ignore
        encoders[column] = label_encoder
        
        
    feature_columns = ["congestion_level", "delay_level", "distance", "time_of_day", "zone_type"]
    X = data[feature_columns]
    y = data["delay_level"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model, encoders



MODEL, ENCODERS = load_and_train_model()