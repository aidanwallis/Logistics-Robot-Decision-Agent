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

def extract_route_features(route, grid, time_of_day):
    steps = len(route) # maybe needs to be -1
    
    # Define distance categories based on the number of steps
    if steps <= 6:
        distance = "short"
    elif steps <= 10:
        distance = "medium"
    else:
        distance = "long"
        
    # look up what symbol is at each cell in the route
    route_cells = [grid[row][column] for row, column in route]
    
    #classify the route based on worst cell type in the route, R > H > normal
    if "R" in route_cells:
        zone_type = "restricted"
    elif "H" in route_cells:
        zone_type = "high_traffic"
    else:
        zone_type = "normal"
        
    # classify congestion level based on number of high traffic cells in the route
    high_traffic_count = route_cells.count("H")
    if high_traffic_count == 0:
        congestion_level = "low"
    elif high_traffic_count <= 2:
        congestion_level = "medium"
    else:
        congestion_level = "high"
    
    
    return time_of_day, congestion_level, distance, zone_type





MODEL, ENCODERS = load_and_train_model()