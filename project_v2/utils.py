import numpy as np
from scipy.optimize import minimize

def calculate_sustainability_metrics(cement, slag, ash, water, sp, coarse, fine):
    # CO2 Factors (kg CO2 per kg) - Estimates
    factors = {
        "cement": 0.9, 
        "slag": 0.05, 
        "ash": 0.02, 
        "water": 0.001, 
        "sp": 1.5, 
        "coarse": 0.008, 
        "fine": 0.01
    }
    co2 = (cement * factors["cement"] + 
           slag * factors[ "slag"] + 
           ash * factors["ash"] + 
           water * factors["water"] + 
           sp * factors["sp"] + 
           coarse * factors["coarse"] + 
           fine * factors["fine"])
    return round(co2, 2)

def calculate_cost_metrics(cement, slag, ash, water, sp, coarse, fine):
    # Rs per kg (Approximate Indian Market Rates)
    costs = {
        "cement": 7.0,      # ~350 per 50kg bag
        "slag": 4.0,
        "ash": 2.5,
        "water": 0.5,
        "sp": 50.0,
        "coarse": 1.2,
        "fine": 1.5
    }
    cost = (cement * costs["cement"] + 
            slag * costs["slag"] + 
            ash * costs["ash"] + 
            water * costs["water"] + 
            sp * costs["sp"] + 
            coarse * costs["coarse"] + 
            fine * costs["fine"])
    return round(cost, 2)

def get_structural_performance_score(strength, cost, co2):
    # Composite score: higher is better
    impact = (cost * 0.01) + (co2 * 0.1)
    if impact == 0: return 0
    return round(strength / impact, 2)

def get_smart_suggestions(strength, cost, co2, age):
    suggestions = []
    if strength < 20 and age >= 28:
        suggestions.append("⚠️ Low strength for structural use. Increase cement content or reduce water-to-cement ratio.")
    if co2 > 500:
        suggestions.append("🌱 High carbon footprint. Consider replacing cement with Fly Ash or GGBS (Slag).")
    if strength > 40:
        suggestions.append("✅ High-performance mix. Suitable for high-rise buildings and bridges.")
    else:
        suggestions.append("ℹ️ Standard strength mix. Suitable for residential foundations and walkways.")
    return suggestions

def optimize_mix(target_strength, model_predict_func):
    # Initial guess: standard mix proportions
    initial_guess = [250, 50, 50, 180, 5, 1000, 800, 28]
    
    # Constraints: ingredients must be positive and within reasonable bounds
    bounds = [
        (100, 500), # Cement
        (0, 300),   # Slag
        (0, 300),   # FlyAsh
        (100, 250), # Water
        (0, 30),    # SP
        (700, 1200),# Coarse
        (500, 1000),# Fine
        (7, 90)     # Age
    ]
    
    def objective(x):
        # Predict strength using the passed function
        pred = model_predict_func(np.array([x]))
        cost = calculate_cost_metrics(*x[:-1])
        # Penalty for deviating from target strength + minimize cost
        return (pred - target_strength)**2 + (cost * 0.05)

    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else initial_guess
