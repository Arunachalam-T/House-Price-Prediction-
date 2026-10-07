"""
Multimodal House Price Prediction Experiment.
Compares Linear Regression, Random Forest, and Gradient Boosting on
Structured Data vs Structured Data + CNN Image Features.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from src.data_loader import ensure_csv, ensure_images, load_dataset
from src.preprocessing import TARGET_COLUMN, build_preprocessor, select_features
from src.cnn_features import extract_image_feature_frame

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0.0) # clip negative prices
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2

def make_model(estimator, X_train, image_columns):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train, image_columns)),
        ("regressor", estimator)
    ])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )

def main():
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    print("Loading dataset...")
    csv_path = ensure_csv(data_dir)
    frame = load_dataset(csv_path)
    
    # We need the images for the multimodal part
    print("Ensuring images are downloaded (this will download ~390MB from Kaggle if absent)...")
    image_dir = ensure_images(data_dir)
    
    # Extract CNN features
    print("Extracting CNN features...")
    cache_path = data_dir / "cnn_features_cache.csv"
    if cache_path.is_file():
        print(f"Loading cached CNN features from {cache_path}...")
        image_frame = pd.read_csv(cache_path, index_col=0)
        image_frame.index = frame.index
    else:
        image_frame = extract_image_feature_frame(frame, image_dir)
        image_frame.to_csv(cache_path)
        print(f"Saved CNN features cache to {cache_path}")
        
    image_columns = image_frame.columns.tolist()
    
    # Prepare features
    X_structured = select_features(frame)
    X_multimodal = pd.concat([X_structured, image_frame], axis=1)
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(float)
    
    # Split data
    row_ids = np.arange(len(frame))
    train_ids, test_ids = train_test_split(row_ids, test_size=0.20, random_state=42)
    
    X_struct_train, X_struct_test = X_structured.iloc[train_ids], X_structured.iloc[test_ids]
    X_multi_train, X_multi_test = X_multimodal.iloc[train_ids], X_multimodal.iloc[test_ids]
    y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
    
    # Baseline models as per project abstract
    models_to_test = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    results = {}
    
    for name, estimator in models_to_test.items():
        print(f"\n========================================")
        print(f"Evaluating: {name}")
        print(f"========================================")
        
        print(f"Training on Structured Data Only...")
        model_struct = make_model(estimator, X_struct_train, [])
        model_struct.fit(X_struct_train, y_train)
        mae_s, rmse_s, r2_s = evaluate_model(model_struct, X_struct_test, y_test)
        
        print(f"Training on Multimodal Data (Structured + CNN)...")
        model_multi = make_model(estimator, X_multi_train, image_columns)
        model_multi.fit(X_multi_train, y_train)
        mae_m, rmse_m, r2_m = evaluate_model(model_multi, X_multi_test, y_test)
        
        results[name] = {
            "Structured": {"MAE": mae_s, "RMSE": rmse_s, "R2": r2_s},
            "Multimodal": {"MAE": mae_m, "RMSE": rmse_m, "R2": r2_m}
        }
        
        print(f"\n{name} Results:")
        print(f"  Structured: MAE=${mae_s:,.2f}, RMSE=${rmse_s:,.2f}, R2={r2_s:.4f}")
        print(f"  Multimodal: MAE=${mae_m:,.2f}, RMSE=${rmse_m:,.2f}, R2={r2_m:.4f}")
        
    out_file = outputs_dir / "multimodal_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiment completed successfully. Results saved to {out_file}")

if __name__ == "__main__":
    main()
