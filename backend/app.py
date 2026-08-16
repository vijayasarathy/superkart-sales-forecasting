"""
SuperKart Sales Forecasting API
--------------------------------
Flask backend serving the tuned XGBoost pipeline trained in the notebook.

Endpoints:
    GET  /                    - health check and API information
    POST /v1/superkart        - predict sales for a single product-store record
    POST /v1/superkartbatch   - predict sales for a CSV file of records
"""

import os
import io

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the serialized pipeline once at startup, not per request
model = joblib.load("superkart_model.joblib")

# Must match the reference year used during training in the notebook.
# Hardcoded deliberately so the deployed model produces identical features
# to those it was trained on, regardless of when the container runs.
REFERENCE_YEAR = 2026

# Columns the model was trained on, in order
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Category",
    "Store_Age",
]


def engineer_features(df):
    """
    Reproduce the exact feature engineering performed in the training notebook.

    This runs on every request. Any change here must be mirrored in the
    notebook, otherwise the features will not match what the model expects.
    """
    df = df.copy()

    # 1. Clean the inconsistent sugar content label
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace(
            "reg", "Regular"
        )

    # 2. Derive the product category from the first two characters of Product_Id
    if "Product_Category" not in df.columns:
        if "Product_Id" not in df.columns:
            raise ValueError(
                "Either 'Product_Id' or 'Product_Category' must be provided"
            )
        df["Product_Category"] = df["Product_Id"].astype(str).str[:2]

    # 3. Derive store age from the establishment year
    if "Store_Age" not in df.columns:
        if "Store_Establishment_Year" not in df.columns:
            raise ValueError(
                "Either 'Store_Establishment_Year' or 'Store_Age' must be provided"
            )
        df["Store_Age"] = REFERENCE_YEAR - pd.to_numeric(
            df["Store_Establishment_Year"], errors="coerce"
        )

    # 4. Keep only the columns the model expects, in the trained order
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return df[MODEL_FEATURES]


@app.route("/", methods=["GET"])
def home():
    """Health check and basic API documentation."""
    return jsonify(
        {
            "message": "SuperKart Sales Forecasting API is running",
            "model": "XGBoost (tuned)",
            "target": "Product_Store_Sales_Total",
            "endpoints": {
                "/v1/superkart": "POST a JSON record to predict sales for one product",
                "/v1/superkartbatch": "POST a CSV file to predict sales for many records",
            },
        }
    )


@app.route("/v1/superkart", methods=["POST"])
def predict_single():
    """Predict total sales for a single product-store combination."""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON body received"}), 400

        input_df = pd.DataFrame([payload])
        features = engineer_features(input_df)
        prediction = model.predict(features)[0]

        return jsonify({"predicted_sales": round(float(prediction), 2)})

    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Prediction failed: {err}"}), 500


@app.route("/v1/superkartbatch", methods=["POST"])
def predict_batch():
    """Predict total sales for every row in an uploaded CSV file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use the 'file' form field."}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        batch_df = pd.read_csv(io.BytesIO(uploaded.read()))
        if batch_df.empty:
            return jsonify({"error": "Uploaded file contains no rows"}), 400

        features = engineer_features(batch_df)
        predictions = model.predict(features)

        results = batch_df.copy()
        # Cast to float64 before rounding - float32 does not round cleanly in JSON
        results["Predicted_Sales"] = predictions.astype(float).round(2)

        return jsonify(
            {
                "rows_processed": len(results),
                "predictions": results.to_dict(orient="records"),
            }
        )

    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Batch prediction failed: {err}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
