"""
Recreate mock models exactly matching the feature arrays the routes send:
  module_a_rf  → 8 features  [Cement, Slag, FlyAsh, Water, Superplasticizer, CoarseAgg, FineAgg, Age]
  module_b_xgb → 11 features [%C, %Mn, %Si, %Cr, %Ni, %Mo, %V, %Al, %Ti, %N, %B]
  module_c_reg → 3 features  [Hardness, Density, Elongation]  → outputs: [Su, Sy]
  module_c_clf → 3 features  [Hardness, Density, Elongation]  → output:  0/1
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

models_dir = Path("models")
models_dir.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# --- Module A: 8 features ---
X_a = np.random.rand(50, 8) * [500, 400, 250, 200, 30, 1100, 900, 365]
y_a = np.random.rand(50) * 80 + 10   # compressive strength 10–90 MPa
rf_a = RandomForestRegressor(n_estimators=20, random_state=42)
rf_a.fit(X_a, y_a)
joblib.dump(rf_a, models_dir / "module_a_rf.pkl")
print("✔ module_a_rf  (8 features) saved")

# --- Module B: 11 features (%C %Mn %Si %Cr %Ni %Mo %V %Al %Ti %N %B) ---
X_b = np.random.rand(50, 11) * [1.5, 3.0, 1.0, 5.0, 5.0, 2.0, 1.0, 0.5, 0.5, 0.05, 0.005]
y_b = np.random.rand(50) * 1500 + 300  # UTS 300–1800 MPa
xgb_b = XGBRegressor(n_estimators=20, random_state=42)
xgb_b.fit(X_b, y_b)
joblib.dump(xgb_b, models_dir / "module_b_xgb.pkl")
print("✔ module_b_xgb (11 features) saved")

# --- Module C Reg: 3 features → 2 outputs (Su, Sy) ---
X_c = np.random.rand(50, 3) * [700, 15, 60]   # Hardness, Density, Elongation
y_cr = np.column_stack([
    np.random.rand(50) * 800 + 200,   # Su
    np.random.rand(50) * 700 + 100,   # Sy
])
rf_cr = MultiOutputRegressor(RandomForestRegressor(n_estimators=20, random_state=42))
rf_cr.fit(X_c, y_cr)
joblib.dump(rf_cr, models_dir / "module_c_reg.pkl")
print("✔ module_c_reg (3 features → 2 outputs) saved")

# --- Module C Clf: 3 features → 0/1 ---
y_cc = (np.random.rand(50) > 0.4).astype(int)
rf_cc = RandomForestClassifier(n_estimators=20, random_state=42)
rf_cc.fit(X_c, y_cc)
joblib.dump(rf_cc, models_dir / "module_c_clf.pkl")
print("✔ module_c_clf (3 features) saved")

print("\nAll mock models saved to ./models/")
