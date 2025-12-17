from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# ==============================
# APP SETUP
# ==============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD MODEL & TOOLS (OPTIMIZED)
# ==============================
MODEL_PATH = "D:/Baymax2/bozo/backend/model/growth_lstm_clean.keras"
SCALER_X_PATH = "D:/Baymax2/bozo/backend/model/scaler_X.pkl"
SCALER_Y_PATH = "D:/Baymax2/bozo/backend/model/scaler_y.pkl"
CONFIG_PATH = "D:/Baymax2/bozo/backend/model/config.pkl"

model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
config = joblib.load(CONFIG_PATH)

SEQ_LEN = config["SEQ_LEN"]
FEATURES = config["X_FEATURES"]
TARGETS = config["Y_TARGETS"]

print(f" Model loaded: {MODEL_PATH}")
print(f" Sequence length: {SEQ_LEN}")
print(f" Features count: {len(FEATURES)}")
print(f" Features: {FEATURES}")


# ==============================
# HELPER: COMPREHENSIVE FEATURE ENGINEERING
# ==============================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ALL feature engineering steps used during training
    Must match EXACTLY with training script
    """
    df = df.sort_values("age_month").reset_index(drop=True)

    # Numeric safety
    for col in ["age_month", "height_cm", "weight_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Velocity features (first derivative)
    df["height_vel"] = df["height_cm"].diff().fillna(0)
    df["weight_vel"] = df["weight_kg"].diff().fillna(0)

    # 2. BMI calculation
    height_m = df["height_cm"] / 100
    height_m[height_m <= 0] = np.nan
    df["BMI"] = df["weight_kg"] / (height_m ** 2)
    df["BMI"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill BMI NaN with median of available values
    bmi_median = df["BMI"].median()
    if pd.isna(bmi_median):
        bmi_median = 16.0  # Default fallback
    df["BMI"].fillna(bmi_median, inplace=True)

    # 3. Acceleration features (second derivative)
    df["height_acc"] = df["height_vel"].diff().fillna(0)
    df["weight_acc"] = df["weight_vel"].diff().fillna(0)

    # 4. Rolling averages (moving average with window=3)
    window_size = 3
    df["height_cm_ma3"] = df["height_cm"].rolling(
        window=window_size, min_periods=1
    ).mean()
    df["weight_kg_ma3"] = df["weight_kg"].rolling(
        window=window_size, min_periods=1
    ).mean()

    # 5. Age-based features
    df["age_years"] = df["age_month"] / 12
    df["age_squared"] = df["age_month"] ** 2
    df["age_sqrt"] = np.sqrt(df["age_month"])

    # 6. Height-to-weight ratio
    df["height_weight_ratio"] = df["height_cm"] / df["weight_kg"]
    df["height_weight_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill ratio NaN with median
    ratio_median = df["height_weight_ratio"].median()
    if pd.isna(ratio_median):
        ratio_median = 2.5  # Default fallback
    df["height_weight_ratio"].fillna(ratio_median, inplace=True)

    # Ensure all features are float32
    feature_cols = [
        "height_vel", "weight_vel", "BMI", "height_acc", "weight_acc",
        "height_cm_ma3", "weight_kg_ma3", "age_years", "age_squared",
        "age_sqrt", "height_weight_ratio"
    ]
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df


# ==============================
# VALIDATION HELPERS
# ==============================
def validate_input(df: pd.DataFrame) -> dict:
    """
    Validate input data for realistic ranges
    Returns dict with 'valid' (bool) and 'error' (str if invalid)
    """
    for i, row in df.iterrows():
        # Height validation (30-200 cm)
        if not (30 <= row["height_cm"] <= 200):
            return {
                "valid": False,
                "error": "Invalid height_cm",
                "detail": f"height_cm at index {i} = {row['height_cm']} cm is not realistic (expected 30-200)"
            }

        # Weight validation (2-80 kg)
        if not (2 <= row["weight_kg"] <= 80):
            return {
                "valid": False,
                "error": "Invalid weight_kg",
                "detail": f"weight_kg at index {i} = {row['weight_kg']} kg is not realistic (expected 2-80)"
            }

        # Age validation (0-240 months = 0-20 years)
        if not (0 <= row["age_month"] <= 240):
            return {
                "valid": False,
                "error": "Invalid age_month",
                "detail": f"age_month at index {i} = {row['age_month']} is out of range (expected 0-240)"
            }

    return {"valid": True}


def check_feature_availability(df: pd.DataFrame, required_features: list) -> dict:
    """
    Check if all required features are present in DataFrame
    """
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        return {
            "valid": False,
            "error": "Missing features after engineering",
            "detail": f"Missing: {missing_features}"
        }
    
    return {"valid": True}


# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.post("/predict")
def predict(data: dict):
    """
    Predict NEXT step (t+1) using optimized LSTM model
    
    Expected input format:
    {
        "history": [
            {"age_month": 12, "height_cm": 75.0, "weight_kg": 9.5},
            {"age_month": 13, "height_cm": 76.0, "weight_kg": 9.8},
            ...
        ]
    }
    
    Returns:
    {
        "predicted_for_age_month": int,
        "predicted_height_cm": float,
        "predicted_weight_kg": float
    }
    """

    try:
        # ------------------------------
        # 1. Input → DataFrame
        # ------------------------------
        if "history" not in data:
            return {
                "error": "Missing 'history' field in request",
                "detail": "Expected format: {'history': [{'age_month': ..., 'height_cm': ..., 'weight_kg': ...}]}"
            }

        df = pd.DataFrame(data["history"])

        # Check minimum data points
        min_required = SEQ_LEN + 1
        if len(df) < min_required:
            return {
                "error": f"Insufficient data points",
                "detail": f"Minimum {min_required} data points required, got {len(df)}"
            }

        # Validate required columns
        required_cols = ["age_month", "height_cm", "weight_kg"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                "error": "Missing required columns",
                "detail": f"Missing: {missing_cols}"
            }

        # ------------------------------
        # 2. Validate Input Ranges
        # ------------------------------
        validation_result = validate_input(df)
        if not validation_result["valid"]:
            return validation_result

        # ------------------------------
        # 3. Feature Engineering
        # ------------------------------
        df = add_features(df)

        # Check if we still have enough data after engineering
        if len(df) < SEQ_LEN:
            return {
                "error": "Insufficient data after feature engineering",
                "detail": f"Need at least {SEQ_LEN} valid records"
            }

        # Verify all required features exist
        feature_check = check_feature_availability(df, FEATURES)
        if not feature_check["valid"]:
            return feature_check

        # ------------------------------
        # 4. Extract Window for Prediction
        # ------------------------------
        window = df.iloc[-SEQ_LEN:].copy()

        # Select features in correct order
        X_raw = window[FEATURES].values

        # Check for NaN or infinite values
        if not np.isfinite(X_raw).all():
            nan_features = [FEATURES[i] for i in range(len(FEATURES)) if not np.isfinite(X_raw[:, i]).all()]
            return {
                "error": "Invalid values detected in features",
                "detail": f"Features with NaN/Inf: {nan_features}"
            }

        # ------------------------------
        # 5. Scale Input Features
        # ------------------------------
        X_scaled = scaler_X.transform(X_raw)
        X_scaled = X_scaled.reshape(1, SEQ_LEN, len(FEATURES))

        # ------------------------------
        # 6. Model Prediction
        # ------------------------------
        y_scaled = model.predict(X_scaled, verbose=0)

        # ------------------------------
        # 7. Inverse Transform to Original Scale
        # ------------------------------
        y_pred = scaler_y.inverse_transform(y_scaled)[0]

        pred_height = float(y_pred[0])
        pred_weight = float(y_pred[1])

        # ------------------------------
        # 8. Post-processing: Growth Constraints
        # ------------------------------
        # Growth should not decrease significantly
        last_height = float(df["height_cm"].iloc[-1])
        last_weight = float(df["weight_kg"].iloc[-1])

        # Height should not decrease
        pred_height = max(pred_height, last_height)

        # Weight can decrease slightly (2% tolerance for measurement variance)
        pred_weight = max(pred_weight, last_weight * 0.98)

        # Sanity check: predictions within realistic bounds
        pred_height = np.clip(pred_height, 30, 200)
        pred_weight = np.clip(pred_weight, 2, 80)

        # ------------------------------
        # 9. Estimate Next Age
        # ------------------------------
        age_diffs = df["age_month"].diff().dropna()
        if len(age_diffs) > 0:
            delta_age = int(age_diffs.median())
        else:
            delta_age = 1  # Default to 1 month

        next_age = int(df["age_month"].iloc[-1] + delta_age)

        # ------------------------------
        # 10. Return Prediction
        # ------------------------------
        return {
            "predicted_for_age_month": next_age,
            "predicted_height_cm": round(pred_height, 2),
            "predicted_weight_kg": round(pred_weight, 2),
            "input_sequence_length": SEQ_LEN,
            "last_recorded_height": round(last_height, 2),
            "last_recorded_weight": round(last_weight, 2)
        }

    except Exception as e:
        # Catch any unexpected errors
        return {
            "error": "Unexpected error during prediction",
            "detail": str(e),
            "type": type(e).__name__
        }


# ==============================
# HEALTH CHECK ENDPOINT
# ==============================
@app.get("/")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model": "Growth LSTM Optimized",
        "version": config.get("MODEL_VERSION", "2.0"),
        "sequence_length": SEQ_LEN,
        "features_count": len(FEATURES),
        "total_parameters": config.get("TOTAL_PARAMS", "N/A")
    }


# ==============================
# MODEL INFO ENDPOINT
# ==============================
@app.get("/model-info")
def model_info():
    """
    Get detailed model information
    """
    return {
        "model_version": config.get("MODEL_VERSION", "2.0"),
        "sequence_length": SEQ_LEN,
        "features": FEATURES,
        "targets": TARGETS,
        "batch_size": config.get("BATCH_SIZE", 32),
        "learning_rate": config.get("LEARNING_RATE", 0.001),
        "total_parameters": config.get("TOTAL_PARAMS", "N/A")
    }
