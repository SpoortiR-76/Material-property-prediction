import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "healthy" in response.json()["status"]
    assert "models_loaded" in response.json()
    assert isinstance(response.json()["models_loaded"], list)

def test_predict_module_a_rf_valid():
    response = client.post(
        "/predict/module_a_rf",
        json={"features": {
            "Cement": 300.0, "Slag": 0.0, "FlyAsh": 0.0,
            "Water": 180.0, "Superplasticizer": 6.0,
            "CoarseAgg": 1000.0, "FineAgg": 750.0, "Age": 28.0
        }}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)
    assert response.json()["model_used"] == "module_a_rf"

def test_predict_module_b_xgb_valid():
    response = client.post(
        "/predict/module_b_xgb",
        json={"features": {
            "%C": 0.25, "%Mn": 1.5, "%Si": 0.3, "%Cr": 0.8, "%Ni": 0.0,
            "%Mo": 0.0, "%V": 0.0, "%Al": 0.03, "%Ti": 0.0,
            "%N": 0.005, "%B": 0.0
        }}
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == "module_b_xgb"
    assert isinstance(response.json()["prediction"], float)

def test_predict_module_c_reg_valid():
    response = client.post(
        "/predict/module_c_reg",
        json={"features": {
            "Hardness": 200.0, "Density": 7.8, "Elongation": 20.0
        }}
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == "module_c_reg"
    assert isinstance(response.json()["prediction"], dict)
    assert "Su" in response.json()["prediction"] and "Sy" in response.json()["prediction"]

def test_predict_invalid_model_name():
    response = client.post("/predict/nonexistent_model", json={"features": {}})
    assert response.status_code == 404
    assert "detail" in response.json()

def test_predict_missing_feature():
    response = client.post("/predict/module_a_rf", json={"features": {"Cement": 300.0}})
    assert response.status_code == 422

def test_predict_invalid_feature_value():
    response = client.post(
        "/predict/module_a_rf",
        json={"features": {
            "Cement": -999, "Water": -100, "Age": -5,
            "Slag": 0, "FlyAsh": 0, "Superplasticizer": 0,
            "CoarseAgg": 1000, "FineAgg": 750
        }}
    )
    assert response.status_code == 422
