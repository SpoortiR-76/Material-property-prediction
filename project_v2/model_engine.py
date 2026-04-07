import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# Relative Paths for independent folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Column Definitions
CONCRETE_COLS = ["Cement", "Slag", "FlyAsh", "Water", "SP", "CoarseAgg", "FineAgg", "Age"]
STEEL_COLS = ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co"]
MECH_COLS = ["E", "G", "mu", "Ro"]
MECH_TARGETS_REG = ["Su", "Sy"]
MECH_TARGET_CLS = "Use"

def load_data(material_type):
    if material_type == "concrete":
        path = os.path.join(DATA_DIR, "Concrete_Data.xls")
        df = pd.read_excel(path)
        df.columns = CONCRETE_COLS + ["Strength"]
        return df, CONCRETE_COLS, "Strength"
    elif material_type == "steel":
        path = os.path.join(DATA_DIR, "steel_strength.csv")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        target = "tensile strength"
        return df, STEEL_COLS, target
    else:
        path = os.path.join(DATA_DIR, "mechanical_material.csv")
        df = pd.read_csv(path)
        return df, MECH_COLS, MECH_TARGETS_REG

def train_model(material_type):
    data_mode = material_type.split('_')[0]
    df, features, target = load_data(data_mode)
    X = df[features]
    
    if material_type == "concrete":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        y = df[target]
    elif material_type == "mechanical":
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        y = df[target]
    elif material_type == "mechanical_clf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y = df[MECH_TARGET_CLS]
    else:
        model = LinearRegression() # for steel
        y = df[target]
    
    model.fit(X, y)
    
    model_file = os.path.join(MODELS_DIR, f"{material_type}_model.pkl")
    joblib.dump(model, model_file)
    
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(features, model.feature_importances_))
    elif hasattr(model, 'estimators_'):
        # For multi-output, average importance across outputs
        imps = [est.feature_importances_ for est in model.estimators_]
        importance = dict(zip(features, np.mean(imps, axis=0)))
    elif hasattr(model, 'coef_'):
        importance = dict(zip(features, np.abs(model.coef_)))
        
    return model, importance

def predict(material_type, input_data):
    model_file = os.path.join(MODELS_DIR, f"{material_type}_model.pkl")
    if not os.path.exists(model_file):
        model, _ = train_model(material_type)
    else:
        model = joblib.load(model_file)
    
    cols = MECH_COLS if "mech" in material_type else (CONCRETE_COLS if material_type == "concrete" else STEEL_COLS)
    input_df = pd.DataFrame([input_data], columns=cols)
    return model.predict(input_df)[0]

def get_feature_importance(material_type):
    _, importance = train_model(material_type)
    return importance
