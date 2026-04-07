import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_sustainability_metrics(cement, slag, ash, water, sp, coarse, fine):
    # Standard CO2 emission factors (kg CO2 per kg material) - Approximate values
    factors = {
        "cement": 0.9,
        "slag": 0.05,
        "ash": 0.02,
        "water": 0.0003,
        "sp": 1.5,
        "coarse": 0.008,
        "fine": 0.005
    }
    
    co2 = (cement * factors["cement"] + 
           slag * factors["slag"] + 
           ash * factors["ash"] + 
           water * factors["water"] + 
           sp * factors["sp"] + 
           coarse * factors["coarse"] + 
           fine * factors["fine"])
    
    return round(co2, 2)

def calculate_cost_metrics(cement, slag, ash, water, sp, coarse, fine):
    # Standard cost factors (Rs per kg) - Approximate Indian Market Rates
    costs = {
        "cement": 7.0,      # ~350 per 50kg bag
        "slag": 4.0,
        "ash": 2.5,
        "water": 0.5,
        "sp": 50.0,
        "coarse": 1.2,
        "fine": 1.5
    }
    
    total_cost = (cement * costs["cement"] + 
                  slag * costs["slag"] + 
                  ash * costs["ash"] + 
                  water * costs["water"] + 
                  sp * costs["sp"] + 
                  coarse * costs["coarse"] + 
                  fine * costs["fine"])
    
    return round(total_cost, 2)

def get_structural_performance_score(strength, cost, co2):
    """
    SPS = Strength / (Normalized Cost * Normalized CO2) 
    A higher score is better (stronger, cheaper, greener).
    Using simple normalization (assuming typical ranges).
    """
    if cost == 0 or co2 == 0:
        return 0
    
    # Simple ratio for demonstration
    score = strength / ( (cost/10) * (co2/100) )
    return round(score, 2)

def get_smart_suggestions(strength, cost, co2, age):
    suggestions = []
    
    if strength < 30:
        suggestions.append("⚠️ Low Strength: Consider increasing cement content or using a higher grade. Ensure curing for at least 28 days.")
    
    if co2 > 500:
        suggestions.append("🌱 High Carbon Footprint: Try replacing part of the cement with Fly Ash or Slag to improve sustainability.")
    
    if cost > 100:
        suggestions.append("💰 High Cost: Review the usage of Superplasticizer. High dosage increases cost significantly.")
    
    if age < 28:
        suggestions.append("⏳ Early Age: Predicted strength is based on early curing. Strength typically increases by 20-30% by day 28.")
        
    if not suggestions:
        suggestions.append("✅ Optimal Mix: This combination provides a good balance of properties.")
        
    return suggestions

def optimize_mix(target_strength, model_predict_func):
    """
    Very simple optimization: 
    Find a mix that minimizes (predicted_strength - target_strength)^2 + (cost / 100)
    """
    # X = [Cement, Slag, FlyAsh, Water, Superplasticizer, CoarseAgg, FineAgg, Age]
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
        # We need a 2D array for the model
        pred = model_predict_func(np.array([x]))
        cost = calculate_cost_metrics(*x[:-1]) # ignore age for cost
        return (pred - target_strength)**2 + (cost * 0.1)

    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        return res.x
    return initial_guess
