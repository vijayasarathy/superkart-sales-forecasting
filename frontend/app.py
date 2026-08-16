"""
SuperKart Sales Forecasting - Streamlit Frontend
-------------------------------------------------
A web interface for the SuperKart sales forecasting API.

Supports two modes:
    - Single Prediction: enter one product-store combination via a form
    - Batch Prediction : upload a CSV and receive predictions for every row
"""

import os
import io

import requests
import pandas as pd
import streamlit as st

# The backend URL is read from an environment variable so the same image works
# both inside a Docker network and against a public Codespace URL.
DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://superkart-backend:7860")

# Valid categories, taken from the training data
PRODUCT_TYPES = [
    "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
    "Dairy", "Baking Goods", "Canned", "Health and Hygiene", "Meat",
    "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods",
    "Breakfast", "Seafood",
]
SUGAR_CONTENT = ["Low Sugar", "Regular", "No Sugar"]
PRODUCT_CATEGORIES = {"FD - Food": "FD", "NC - Non-Consumable": "NC", "DR - Drinks": "DR"}

# The four stores in the dataset, with their fixed attributes
STORES = {
    "OUT001 - Supermarket Type1 (Tier 2, High)": {
        "Store_Id": "OUT001", "Store_Establishment_Year": 1987,
        "Store_Size": "High", "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type1",
    },
    "OUT002 - Food Mart (Tier 3, Small)": {
        "Store_Id": "OUT002", "Store_Establishment_Year": 1998,
        "Store_Size": "Small", "Store_Location_City_Type": "Tier 3",
        "Store_Type": "Food Mart",
    },
    "OUT003 - Departmental Store (Tier 1, Medium)": {
        "Store_Id": "OUT003", "Store_Establishment_Year": 1999,
        "Store_Size": "Medium", "Store_Location_City_Type": "Tier 1",
        "Store_Type": "Departmental Store",
    },
    "OUT004 - Supermarket Type2 (Tier 2, Medium)": {
        "Store_Id": "OUT004", "Store_Establishment_Year": 2009,
        "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
    },
}

st.set_page_config(page_title="SuperKart Sales Forecasting", page_icon=None, layout="wide")

st.title("SuperKart Sales Forecasting")
st.write(
    "Predict the total sales revenue a product will generate in a given store. "
    "Powered by a tuned XGBoost model trained on 8,763 historical sales records."
)

# Sidebar - backend configuration and connection status
st.sidebar.header("Backend Configuration")
backend_url = st.sidebar.text_input("API URL", value=DEFAULT_BACKEND).rstrip("/")

if st.sidebar.button("Test connection"):
    try:
        r = requests.get(backend_url + "/", timeout=10)
        if r.status_code == 200:
            st.sidebar.success("Connected")
            st.sidebar.json(r.json())
        else:
            st.sidebar.error(f"Status {r.status_code}")
    except Exception as err:
        st.sidebar.error(f"Cannot reach backend: {err}")

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])


# ----------------------------- Single prediction -----------------------------
with tab_single:
    st.subheader("Predict sales for one product")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Product details**")
        product_category_label = st.selectbox("Product Category", list(PRODUCT_CATEGORIES.keys()))
        product_type = st.selectbox("Product Type", PRODUCT_TYPES)
        sugar_content = st.selectbox("Sugar Content", SUGAR_CONTENT)
        product_weight = st.number_input(
            "Product Weight", min_value=1.0, max_value=30.0, value=12.5, step=0.1
        )
        product_mrp = st.number_input(
            "Product MRP", min_value=10.0, max_value=350.0, value=147.0, step=1.0
        )
        allocated_area = st.number_input(
            "Allocated Display Area (ratio)",
            min_value=0.001, max_value=0.5, value=0.068, step=0.001, format="%.3f"
        )

    with col2:
        st.markdown("**Store details**")
        store_label = st.selectbox("Store", list(STORES.keys()))
        store = STORES[store_label]

        st.write("")
        st.info(
            f"**Store Type:** {store['Store_Type']}  \n"
            f"**Size:** {store['Store_Size']}  \n"
            f"**City Tier:** {store['Store_Location_City_Type']}  \n"
            f"**Established:** {store['Store_Establishment_Year']}"
        )

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Category": PRODUCT_CATEGORIES[product_category_label],
            "Product_Type": product_type,
            "Product_Sugar_Content": sugar_content,
            "Product_Weight": product_weight,
            "Product_MRP": product_mrp,
            "Product_Allocated_Area": allocated_area,
            "Store_Establishment_Year": store["Store_Establishment_Year"],
            "Store_Size": store["Store_Size"],
            "Store_Location_City_Type": store["Store_Location_City_Type"],
            "Store_Type": store["Store_Type"],
        }

        try:
            response = requests.post(backend_url + "/v1/superkart", json=payload, timeout=30)
            if response.status_code == 200:
                prediction = response.json()["predicted_sales"]
                st.success(f"Predicted Sales: {prediction:,.2f}")
                
            else:
                st.error(f"API returned {response.status_code}: {response.json().get('error')}")
        except Exception as err:
            st.error(f"Request failed: {err}")


# ----------------------------- Batch prediction ------------------------------
with tab_batch:
    st.subheader("Predict sales for many products")
    st.write(
        "Upload a CSV with the same columns as the SuperKart dataset. "
        "Every row will receive a predicted sales figure."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        preview = pd.read_csv(uploaded_file)
        st.write(f"File loaded: **{len(preview)} rows**, {len(preview.columns)} columns")
        st.dataframe(preview.head())

        if st.button("Run Batch Prediction", type="primary"):
            uploaded_file.seek(0)
            try:
                response = requests.post(
                    backend_url + "/v1/superkartbatch",
                    files={"file": uploaded_file},
                    timeout=180,
                )
                if response.status_code == 200:
                    result = response.json()
                    results_df = pd.DataFrame(result["predictions"])

                    st.success(f"Predicted sales for {result['rows_processed']} rows")
                    st.dataframe(results_df)

                    total = results_df["Predicted_Sales"].sum()
                    average = results_df["Predicted_Sales"].mean()

                    c1, c2 = st.columns(2)
                    c1.metric("Total predicted sales", f"{total:,.2f}")
                    c2.metric("Average per product", f"{average:,.2f}")

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions as CSV",
                        data=csv_bytes,
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API returned {response.status_code}: {response.json().get('error')}")
            except Exception as err:
                st.error(f"Request failed: {err}")
