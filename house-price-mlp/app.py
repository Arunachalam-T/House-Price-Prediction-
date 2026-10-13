"""
Streamlit Web UI for House Price Prediction (Multimodal MLP + ResNet18 CNN).
Run with: streamlit run app.py
"""
import tempfile
from pathlib import Path
import joblib
import pandas as pd
from PIL import Image
import streamlit as st

from src.predict import predict_price

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
DATA_PATH = ROOT / "data" / "socal2.csv"
SAMPLE_PICS_DIR = ROOT / "data" / "socal_pics"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 1.8rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        margin-top: 1.5rem;
    }
    .price-label {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .price-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        st.error(f"Trained model not found at {MODEL_PATH}. Please run `python main.py --train` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_cities():
    if DATA_PATH.is_file():
        df = pd.read_csv(DATA_PATH, usecols=["citi"])
        cities = df["citi"].dropna().value_counts().index.tolist()
        return cities
    return ["San Diego, CA", "Los Angeles, CA", "Lancaster, CA", "Irvine, CA"]


st.markdown('<div class="main-title">🏡 Multimodal House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting house price using <b>Structured Housing Specs</b> + <b>Exterior Photo (ResNet-18 CNN)</b></div>', unsafe_allow_html=True)

bundle = load_model()
cities = load_cities()
OUTPUTS_DIR = ROOT / "outputs"

tab_predict, tab_graphs = st.tabs(["🏡 Live Prediction", "📊 Model Analytics & Graphs"])

with tab_predict:
    col_left, col_right = st.columns([1.1, 1.2], gap="large")

    with col_left:
        st.subheader("📋 House Specifications")
        
        city = st.selectbox("City / Location", options=cities, index=0)
        
        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            bed = st.number_input("Bedrooms", min_value=1, max_value=12, value=3, step=1)
            bath = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=1.0)
        with col_spec2:
            sqft = st.number_input("Square Footage (sqft)", min_value=300, max_value=15000, value=1800, step=50)

        st.markdown("---")
        predict_btn = st.button("✨ Predict House Price", use_container_width=True, type="primary")

    with col_right:
        st.subheader("📷 House Exterior Photo")
        
        image_option = st.radio(
            "Choose image source:",
            ["Upload an image", "Select a sample dataset image", "No image (use average appearance)"],
            horizontal=True
        )
        
        selected_image = None
        temp_img_path = None
        
        if image_option == "Upload an image":
            uploaded_file = st.file_uploader("Upload House Exterior Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                st.image(selected_image, caption="Uploaded House Photo", use_container_width=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    selected_image.convert("RGB").save(tmp.name)
                    temp_img_path = Path(tmp.name)
                    
        elif image_option == "Select a sample dataset image":
            sample_ids = [0, 10, 50, 100, 200, 500, 1000]
            chosen_id = st.selectbox("Sample Image ID", sample_ids)
            candidate_path = SAMPLE_PICS_DIR / f"{chosen_id}.jpg"
            if candidate_path.is_file():
                selected_image = Image.open(candidate_path)
                st.image(selected_image, caption=f"Dataset Photo #{chosen_id}", use_container_width=True)
                temp_img_path = candidate_path
            else:
                st.info("Sample image not found in local directory.")

    # Prediction Execution
    if predict_btn:
        with st.spinner("Analyzing house features and processing photo via ResNet-18 CNN..."):
            house_input = {
                "citi": str(city),
                "bed": int(bed),
                "bath": float(bath),
                "sqft": float(sqft),
            }
            
            try:
                predicted_price = predict_price(bundle, house_input, image_path=temp_img_path)
                
                with col_left:
                    st.markdown(f"""
                    <div class="price-card">
                        <div class="price-label">Estimated Valuation</div>
                        <div class="price-value">${predicted_price:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"Inputs: {bed} bed, {bath:g} bath, {sqft:,} sqft in {city}. Photo: {'Attached (ResNet-18 extracted)' if temp_img_path else 'None (imputed)'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab_graphs:
    st.subheader("📈 Model Performance & Evaluation Metrics")
    
    # Load and display metrics if present
    metrics_file = OUTPUTS_DIR / "metrics.json"
    if metrics_file.is_file():
        import json
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        test_m = metrics_data.get("test_metrics", {})
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("R² Score (Accuracy)", f"{test_m.get('r2', 0):.4f}")
        m_col2.metric("Mean Absolute Error (MAE)", f"${test_m.get('mae_usd', 0):,.0f}")
        m_col3.metric("Root Mean Sq Error (RMSE)", f"${test_m.get('rmse_usd', 0):,.0f}")
        m_col4.metric("Dataset Size", f"{metrics_data.get('rows_after_target_validation', 15474):,} Homes")
    
    st.markdown("---")

    # Architecture Comparison Section
    comp_chart = OUTPUTS_DIR / "model_comparison.png"
    if comp_chart.is_file():
        st.markdown("### 🏆 Model Comparison: Linear Regression vs Random Forest vs Gradient Boosting vs MLP")
        st.image(str(comp_chart), caption="Architectural Benchmarking: Comparing Models on Structured Only vs. Multimodal Data", use_container_width=True)
        st.caption("Demonstrates the evolutionary stages of the project: evaluating traditional baselines against our multimodal pipeline.")
        st.markdown("---")
    
    st.markdown("### 🔬 In-Depth Diagnostic Plots (Proposed Multimodal MLP)")
    
    # Grid of graphs
    g_col1, g_col2 = st.columns(2)

    
    with g_col1:
        actual_vs_pred = OUTPUTS_DIR / "actual_vs_predicted.png"
        if actual_vs_pred.is_file():
            st.image(str(actual_vs_pred), caption="Actual vs. Predicted House Prices", use_container_width=True)
            st.caption("Points closely clustered around the 45° diagonal line indicate high alignment between true prices and model predictions.")
            
        err_dist = OUTPUTS_DIR / "error_distribution.png"
        if err_dist.is_file():
            st.image(str(err_dist), caption="Distribution of Prediction Errors (Residuals)", use_container_width=True)
            st.caption("A symmetric bell curve centered at zero demonstrates an unbiased model without persistent over- or under-estimation.")
            
        price_dist = OUTPUTS_DIR / "price_distribution.png"
        if price_dist.is_file():
            st.image(str(price_dist), caption="Raw Target Distribution vs. Log1p Transformation", use_container_width=True)
            st.caption("Log transformation removes right-skewness in house prices, stabilizing neural network gradient descent.")

    with g_col2:
        residual_plot = OUTPUTS_DIR / "residual_plot.png"
        if residual_plot.is_file():
            st.image(str(residual_plot), caption="Residuals vs. Predicted Price", use_container_width=True)
            st.caption("Shows error spread across different price tiers; verifies homoscedasticity across the housing price spectrum.")
            
        loss_curve = OUTPUTS_DIR / "loss_curve.png"
        if loss_curve.is_file():
            st.image(str(loss_curve), caption="MLP Training & Validation Loss Curve", use_container_width=True)
            st.caption("Demonstrates smooth optimizer convergence and early stopping without catastrophic overfitting.")

