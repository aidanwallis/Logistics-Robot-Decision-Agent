# CPSC 481 - Logistics Robot Decision Agent ml.py
# File Author: Nicholas Reardon

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = ["congestion_level", "distance", "time_of_day", "zone_type"]

def load_and_train_model():
    """
    Loads the training dataset, encodes all text columns to integers,
    and trains a Decision Tree classifier to predict delay_level.

    Returns:
        model: trained DecisionTreeClassifier
        encoders: dict mapping column name to its fitted LabelEncoder
    """
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
    """
    Converts a route into the four text features the ML model expects.

    Args:
        route: list of (row, col) tuples from search.py
        grid: 2D warehouse grid (list of lists of symbols)
        time_of_day: user-supplied string - morning, afternoon, evening, or night

    Returns:
        tuple: (time_of_day, congestion_level, distance, zone_type) as strings
    """
    steps = len(route)
    
    if steps <= 6:
        distance = "short"
    elif steps <= 10:
        distance = "medium"
    else:
        distance = "long"
        
    route_cells = [grid[row][column] for row, column in route]
    
    if "R" in route_cells:
        zone_type = "restricted"
    elif "H" in route_cells:
        zone_type = "high_traffic"
    else:
        zone_type = "normal"
        
    high_traffic_count = route_cells.count("H")
    if high_traffic_count == 0:
        congestion_level = "low"
    elif high_traffic_count <= 2:
        congestion_level = "medium"
    else:
        congestion_level = "high"
    
    return time_of_day, congestion_level, distance, zone_type


def predict_delay(time_of_day, zone_type, congestion_level, distance):
    """
    Predicts the delay level for a route given its four descriptive features.

    Args:
        time_of_day: morning, afternoon, evening, or night
        zone_type: normal, high_traffic, or restricted
        congestion_level: low, medium, or high
        distance: short, medium, or long

    Returns:
        str: predicted delay level - low, medium, or high
    """
    encoded_time = ENCODERS["time_of_day"].transform([time_of_day])[0]
    encoded_zone = ENCODERS["zone_type"].transform([zone_type])[0]
    encoded_congestion = ENCODERS["congestion_level"].transform([congestion_level])[0]
    encoded_distance = ENCODERS["distance"].transform([distance])[0]
    
    input_features = pd.DataFrame(
        [[encoded_time, encoded_zone, encoded_congestion, encoded_distance]],
        columns=FEATURE_COLUMNS
    )
    
    numeric_prediction = MODEL.predict(input_features)[0]
    predicted_label = ENCODERS["delay_level"].inverse_transform([numeric_prediction])[0]
    
    return predicted_label



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
