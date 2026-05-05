# CPSC 481 - Logistics Robot Decision Agent ml.py
# File Author: Nicholas Reardon

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = ["congestion_level", "distance", "time_of_day", "zone_type"]

def load_and_train_model():
    data = pd.read_csv('data/robot_delay_data.csv')
    
    encoders = {}
    for column in data.columns:
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column]) # type: ignore
        encoders[column] = label_encoder
        
        
    X = data[FEATURE_COLUMNS]
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

def predict_delay(time_of_day, zone_type, congestion_level, distance):
    encoded_time = ENCODERS["time_of_day"].transform([time_of_day])[0]
    encoded_zone = ENCODERS["zone_type"].transform([zone_type])[0]
    encoded_congestion = ENCODERS["congestion_level"].transform([congestion_level])[0]
    encoded_distance = ENCODERS["distance"].transform([distance])[0]
    
    input_features = pd.DataFrame([[encoded_time, encoded_zone, encoded_congestion, encoded_distance]],columns=FEATURE_COLUMNS)   
    
    numeric_prediction = MODEL.predict(input_features)[0]
    
    predicted_label = ENCODERS["delay_level"].inverse_transform([numeric_prediction])[0]
    
    return predicted_label


def get_delay_probability(time_of_day):

    delay_probability_tabel = {
        "morning": 0.25,
        "afternoon": 0.45,
        "evening": 0.7,
        "night": 0.2
    }

    # default to 0.5 if time_of_day is not recognized
    return delay_probability_tabel.get(time_of_day, 0.5) 


MODEL, ENCODERS = load_and_train_model()

# SELF-TEST
if __name__ == "__main__":

    # Test 1: evening + high_traffic + high congestion + short distance -> expect HIGH delay
    result1 = predict_delay("evening", "high_traffic", "high", "short")
    print("Test 1 (evening/high_traffic/high/short):", result1)
    print("  Expected: high")

    # Test 2: morning + normal + low congestion + short distance -> expect LOW delay
    result2 = predict_delay("morning", "normal", "low", "short")
    print("Test 2 (morning/normal/low/short):", result2)
    print("  Expected: low")

    # Test 3: verify delay probability lookup
    print("Evening delay probability:", get_delay_probability("evening"))
    print("  Expected: 0.7")

    print("Morning delay probability:", get_delay_probability("morning"))
    print("  Expected: 0.25")