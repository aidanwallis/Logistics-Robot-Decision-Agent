# CPSC 481 - Logistics Robot Decision Agent reasoning.py
# File Author: Noah Scott

"""
Logical reasoning and uncertainty scoring for candidate robot routes.

This module is intentionally independent from search.py and ml.py. The search
module supplies candidate paths, the ML module supplies route features and a
predicted delay, and this module decides which route should be accepted,
deprioritized, rejected, or selected.
"""

DISTANCE_COST = {
    "short": 1,
    "medium": 2,
    "long": 3,
}

DELAY_COST = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

CONGESTION_COST = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

ZONE_COST = {
    "normal": 0,
    "high_traffic": 2,
    "restricted": 99,
}

BASE_HIGH_DELAY_PROBABILITY = {
    "morning": 0.15,
    "afternoon": 0.30,
    "evening": 0.45,
    "night": 0.20,
}

PREDICTED_DELAY_PROBABILITY = {
    "low": 0.15,
    "medium": 0.45,
    "high": 0.75,
}


def get_delay_probability(
    time_of_day,
    predicted_delay=None,
    zone_type=None,
    congestion_level=None,
    distance=None,
):
    """
    Estimate P(delay = high) using a simple weighted probability score.

    The time of day is the base probability. The ML prediction and route
    conditions then adjust the probability so uncertainty affects the final
    route decision.
    """
    probability = BASE_HIGH_DELAY_PROBABILITY.get(time_of_day, 0.30)

    if predicted_delay in PREDICTED_DELAY_PROBABILITY:
        model_probability = PREDICTED_DELAY_PROBABILITY[predicted_delay]
        probability = (probability * 0.40) + (model_probability * 0.60)

    if zone_type == "high_traffic":
        probability += 0.10
    elif zone_type == "restricted":
        probability += 0.20

    if congestion_level == "medium":
        probability += 0.05
    elif congestion_level == "high":
        probability += 0.15

    if distance == "long":
        probability += 0.10

    return round(min(probability, 0.95), 2)


def apply_decision_rules(zone_type, congestion_level, distance, predicted_delay):
    """
    Apply logical decision rules to a route.

    Returns:
        tuple: (status, reasons, penalty)
            status is one of rejected, deprioritized, accepted, or preferred.
            penalty is added to the route score when the route is not rejected.
    """
    reasons = []
    penalty = 0
    rejected = False
    preferred = False

    if zone_type == "restricted":
        rejected = True
        reasons.append("Rule 1: route enters a restricted zone")

    if predicted_delay == "high" and distance == "long":
        rejected = True
        reasons.append("Rule 2: predicted delay is high and distance is long")

    if congestion_level == "high":
        penalty += 20
        reasons.append("Rule 3: high congestion deprioritizes this route")

    if zone_type == "high_traffic" and predicted_delay != "low":
        penalty += 15
        reasons.append("Rule 4: high-traffic zone with delay risk is less safe")

    if congestion_level == "medium" and distance == "short":
        reasons.append("Rule 5: medium congestion is allowed because route is short")

    if zone_type == "normal" and predicted_delay == "low":
        preferred = True
        reasons.append("Rule 6: normal zone with low predicted delay is preferred")

    if rejected:
        return "rejected", reasons, penalty

    if penalty > 0:
        return "deprioritized", reasons, penalty

    if preferred:
        return "preferred", reasons, penalty

    reasons.append("No rejection rule triggered")
    return "accepted", reasons, penalty


def calculate_weighted_score(
    distance,
    predicted_delay,
    congestion_level,
    zone_type,
    delay_probability,
    rule_penalty=0,
):
    """
    Calculate a lower-is-better route score.

    The score combines route length, ML delay prediction, congestion, zone
    safety, and probability of high delay.
    """
    distance_score = DISTANCE_COST.get(distance, 2) * 10
    delay_score = DELAY_COST.get(predicted_delay, 2) * 15
    congestion_score = CONGESTION_COST.get(congestion_level, 2) * 8
    zone_score = ZONE_COST.get(zone_type, 5)
    probability_score = delay_probability * 30

    return round(
        distance_score
        + delay_score
        + congestion_score
        + zone_score
        + probability_score
        + rule_penalty,
        2,
    )


def evaluate_route(
    route_id,
    path,
    time_of_day,
    zone_type,
    congestion_level,
    distance,
    predicted_delay,
):
    """
    Evaluate one candidate route using rules and probability scoring.
    """
    status, reasons, rule_penalty = apply_decision_rules(
        zone_type,
        congestion_level,
        distance,
        predicted_delay,
    )
    delay_probability = get_delay_probability(
        time_of_day,
        predicted_delay,
        zone_type,
        congestion_level,
        distance,
    )

    if status == "rejected":
        score = None
    else:
        score = calculate_weighted_score(
            distance,
            predicted_delay,
            congestion_level,
            zone_type,
            delay_probability,
            rule_penalty,
        )

    return {
        "route_id": route_id,
        "path": path,
        "status": status,
        "reasons": reasons,
        "score": score,
        "delay_probability": delay_probability,
        "time_of_day": time_of_day,
        "zone_type": zone_type,
        "congestion_level": congestion_level,
        "distance": distance,
        "predicted_delay": predicted_delay,
    }


def choose_best_route(route_evaluations):
    """
    Choose the accepted route with the lowest weighted score.
    """
    viable_routes = [
        route
        for route in route_evaluations
        if route["status"] != "rejected" and route["score"] is not None
    ]

    if not viable_routes:
        return None

    return min(viable_routes, key=lambda route: route["score"])


def evaluate_candidate_routes(paths, grid, time_of_day, feature_extractor, delay_predictor):
    """
    Evaluate every route after search and ML have produced their outputs.

    Args:
        paths: list of (route_id, path) tuples
        grid: warehouse grid
        time_of_day: scenario time
        feature_extractor: usually ml.extract_route_features
        delay_predictor: usually ml.predict_delay

    Returns:
        tuple: (route_evaluations, best_route)
    """
    route_evaluations = []

    for route_id, path in paths:
        route_time, congestion_level, distance, zone_type = feature_extractor(
            path,
            grid,
            time_of_day,
        )
        predicted_delay = delay_predictor(
            route_time,
            zone_type,
            congestion_level,
            distance,
        )
        route_evaluations.append(
            evaluate_route(
                route_id,
                path,
                route_time,
                zone_type,
                congestion_level,
                distance,
                predicted_delay,
            )
        )

    return route_evaluations, choose_best_route(route_evaluations)


def print_route_evaluation(route):
    """
    Print one route's reasoning output in a demo-friendly format.
    """
    print(f"\n{route['route_id']}:")
    print(f"  Path:                    {route['path']}")
    print(f"  Distance:                {route['distance']}")
    print(f"  Zone type:               {route['zone_type']}")
    print(f"  Congestion:              {route['congestion_level']}")
    print(f"  Predicted delay:         {route['predicted_delay']}")
    print(f"  P(delay = high):         {route['delay_probability']}")
    print(f"  Reasoning status:        {route['status']}")

    if route["score"] is None:
        print("  Weighted score:          rejected")
    else:
        print(f"  Weighted score:          {route['score']}")

    for reason in route["reasons"]:
        print(f"  - {reason}")


def print_final_decision(route_evaluations, best_route):
    """
    Print all route evaluations and the final selected route.
    """
    for route in route_evaluations:
        print_route_evaluation(route)

    if best_route is None:
        print("\nFinal decision: No safe route selected.")
        print("All candidate routes were rejected by the reasoning rules.")
        return

    print(f"\nFinal decision: {best_route['route_id']}")
    print(f"Selected path:  {best_route['path']}")
    print(f"Reason: lowest accepted weighted score ({best_route['score']}).")
