import joblib
import os
import pathlib
from pathlib import Path
from utils.logger import system_logger, error_logger

VALID_MODEL_NAMES = [
    "module_a_rf",
    "module_b_xgb",
    "module_c_clf",
    "module_c_reg"
]

MODEL_METADATA = {
    "module_a_rf": {
        "description": "Random Forest — Concrete Compressive Strength",
        "task": "regression",
        "target": "Compressive Strength (MPa)",
        "mae": 3.4264,
        "r2": 0.9119,
        "accuracy": None,
        "features": ["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer", "CoarseAgg", "FineAgg", "Age"]
    },
    "module_b_xgb": {
        "description": "XGBoost — Steel Ultimate Tensile Strength",
        "task": "regression",
        "target": "UTS (MPa)",
        "mae": 69.4138,
        "r2": 0.8915,
        "accuracy": None,
        "features": ["%C", "%Mn", "%Si", "%Cr", "%Ni", "%Mo", "%Cu", "%V", "%Al", "%Ti", "%N", "%B", "Reduction_Ratio"]
    },
    "module_c_reg": {
        "description": "MultiOutput Random Forest — Materials Su & Sy Regression",
        "task": "multi-output regression",
        "target": "Su (MPa), Sy (MPa)",
        "mae": {"Su": 170.2137, "Sy": 165.9451},
        "r2": {"Su": 0.4180, "Sy": 0.2509},
        "accuracy": None,
        "features": ["E", "G", "mu", "Ro"]
    },
    "module_c_clf": {
        "description": "Random Forest Classifier — Material Use Classification",
        "task": "classification",
        "target": "Use (Not Used / Used)",
        "mae": None,
        "r2": None,
        "accuracy": 0.9267,
        "features": ["E", "G", "mu", "Ro"]
    }
}

class ModelStore:
    def __init__(self):
        self._models = {}
        self.load_all_models()

    def load_all_models(self):
        system_logger.info("Loading all ML models...")
        models_dir_path = Path(os.getenv("MODELS_DIR", "models"))
        
        for model_name in VALID_MODEL_NAMES:
            path = models_dir_path / f"{model_name}.pkl"
            try:
                model = joblib.load(path)
                self._models[model_name] = model
                system_logger.info(f"Loaded {model_name} from {path}")
            except FileNotFoundError:
                error_logger.error(f"FileNotFoundError: Could not load {model_name} from {path}")
                self._models[model_name] = None
            except Exception as e:
                error_logger.error(f"Exception loading {model_name} from {path}: {e}", exc_info=True)
                self._models[model_name] = None
        
        loaded_count = len(self.get_loaded_models())
        system_logger.info(f"Model loading complete. {loaded_count}/{len(VALID_MODEL_NAMES)} models loaded.")

    def get_model(self, model_name: str):
        if model_name not in VALID_MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}. Valid: {VALID_MODEL_NAMES}")
        if self._models.get(model_name) is None:
            raise RuntimeError(f"Model '{model_name}' failed to load at startup.")
        return self._models[model_name]

    def is_loaded(self, model_name: str) -> bool:
        return self._models.get(model_name) is not None

    def get_loaded_models(self) -> list[str]:
        return [k for k, v in self._models.items() if v is not None]

    def get_metadata(self, model_name: str) -> dict:
        return MODEL_METADATA.get(model_name, {})

model_store = ModelStore()
