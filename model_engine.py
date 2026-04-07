import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Paths based on project structure
DATA_DIR = r"C:\Users\HP\Downloads\material_pp\material_ml_project\data\raw"
MODELS_DIR = r"c:\Users\HP\Downloads\Materials_ML\models"

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
    df, features, target = load_data(material_type)
    X, y = df[features], df[target]
    
    if material_type == "concrete":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif material_type == "mechanical":
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    elif material_type == "mechanical_clf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y = df[MECH_TARGET_CLS]
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    
    model_file = os.path.join(MODELS_DIR, f"{material_type}_model.pkl")
    joblib.dump(model, model_file)
    
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(features, model.feature_importances_))
    elif hasattr(model, 'estimators_'):
        # For multi-output, average importance
        imps = [est.feature_importances_ for est in model.estimators_]
        importance = dict(zip(features, np.mean(imps, axis=0)))
        
    return model, importance

def predict(material_type, input_data):
    model_file = os.path.join(MODELS_DIR, f"{material_type}_model.pkl")
    if not os.path.exists(model_file):
        model, _ = train_model(material_type)
    else:
        model = joblib.load(model_file)
    
    input_df = pd.DataFrame([input_data], columns=MECH_COLS if "mech" in material_type else (CONCRETE_COLS if material_type == "concrete" else STEEL_COLS))
    return model.predict(input_df)[0]

def get_feature_importance(material_type):
    _, importance = train_model(material_type)
    return importance

if __name__ == "__main__":
    # Test training
    c_mod, c_imp = train_model("concrete")
    s_mod, s_imp = train_model("steel")
    print("Concrete Importance:", c_imp)
    print("Steel Importance:", s_imp)
