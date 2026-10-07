# House Price Prediction using Multilayer Perceptron (MLP)

A reproducible, beginner-friendly supervised regression project for estimating Southern California house prices with **`sklearn.neural_network.MLPRegressor`**. The final predictor is an MLP; no tree model or deep-learning framework is used.

## 1. Objective

Use the property's city, bedroom count, bathroom count, and living area to predict its listed price in US dollars. The program supports training, held-out evaluation, visual reports, and an interactive command-line prediction.

## 2. Dataset source and observed structure

Source: [House Prices and Images - SoCal on Kaggle](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal).

I inspected the actual public Kaggle CSV and archive before selecting columns. The archive contained `socal2.csv` and 15,474 JPEGs under `socal2/socal_pics/`. The CSV contained **15,474 records and these eight columns**:

| Column | Observed meaning / treatment |
|---|---|
| `image_id` | Integer image filename key; excluded as a predictive input |
| `street` | Address text; 12,401 unique values, excluded to avoid a very sparse/high-cardinality address feature |
| `citi` | City label (415 unique cities); used as a categorical feature |
| `n_citi` | Label-encoded version of city; excluded because `citi` is retained and the integer codes have no meaningful numeric distance |
| `bed` | Bedrooms; numeric input |
| `bath` | Bathrooms; numeric input |
| `sqft` | Living area in square feet; numeric input |
| `price` | **Target**, USD; excluded from inputs |

The inspected CSV had **no missing cells**. The code still imputes missing predictor values to support modified or future data. It rejects rows with missing/non-positive targets. There are no exact duplicate rows in the inspected CSV. Price ranges from $195,000 to $2,000,000. Its sample skewness was about **1.349** raw and **0.038** after `log1p`, so the model is fit on `log1p(price)` and predictions are inverse transformed with `expm1`. All reported evaluation metrics are calculated in original USD units.

The inspected archive had 15,474 unique `image_id` values and each had an exact matching `socal2/socal_pics/<image_id>.jpg` file. Thus the image linkage is reliable and is based on the ID, not row position.

## 3. Features used

Default model features:

- `citi` (categorical)
- `bed` (numeric)
- `bath` (numeric)
- `sqft` (numeric)

`image_id`, `street`, and `n_citi` are intentionally omitted for the reasons above. This keeps the ordinary user interface practical: a user can describe a house without providing a unique address, image identifier, or encoded city number.

### Optional handcrafted image features

`src/image_features.py` implements a lightweight, classical computer-vision feature path: resize each linked photo to 64×64, calculate RGB/color statistics and a Histogram of Oriented Gradients (HOG), then reduce HOG to 24 dimensions with PCA inside the sklearn preprocessing pipeline. It uses Pillow and scikit-image—**not** a pretrained network, TensorFlow, or PyTorch.

Images are not bundled because the Kaggle image collection is about 390 MB. I ran an image-vs-tabular ablation using the same 65/15/20 split and the same 4-candidate, 3-fold randomized MLP search. The tabular model achieved test MAE **$135,549**, RMSE **$224,528**, R² **0.658**. Adding color statistics + HOG/PCA image descriptors resulted in MAE **$181,823**, RMSE **$280,911**, R² **0.465** (MAE about **34.1% worse**). Some image-search fits also reached the configured 300-iteration cap. For this handcrafted feature setup and tuning budget, the exterior photos did not improve held-out prediction, so the delivered/default final model uses tabular features. This is not evidence that all possible image methods would fail. The measured comparison is in `outputs/image_ablation_metrics.json`.

The classical image path remains available for further experiments: use `python main.py --train --use-images`; if `data/socal_pics/` is missing, the script downloads the public Kaggle archive and extracts the linked JPEGs. Image-enabled training and prediction are heavier, and prediction works best when a new-house photo is supplied (`--image`). If no new image is supplied, the preprocessing pipeline imputes median image features. The image path uses no pretrained model or deep-learning framework.

## 4. Why MLPRegressor

This is an academic supervised-learning regression project whose required syllabus method is a multilayer perceptron. The implementation uses `sklearn.neural_network.MLPRegressor` as the **main and final model**. It learns nonlinear relationships among city, room counts, and floor area. MLP training is sensitive to feature scale, so numeric features are standardized and city categories are one-hot encoded.

## 5. Preprocessing and leakage control

- Numerical inputs: median imputation, then `StandardScaler`.
- Categorical inputs: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore", min_frequency=5)`.
- Optional HOG descriptors: median imputation, scaling, and PCA to 24 components.
- Preprocessing is assembled with `ColumnTransformer` and `Pipeline`. It is fit inside each cross-validation training fold; validation/test data do not fit imputers, scaling, city categories, or PCA.
- Fixed split: 65% train, 15% validation, 20% test (random seed 42). Randomized cross-validation is performed only on the training portion; early stopping uses an internal training fold validation fraction.

## 6. Model architecture and tuning

The estimator is a `TransformedTargetRegressor` wrapping a preprocessing `Pipeline` and `MLPRegressor`:

- ReLU activation, Adam solver
- 300 maximum iterations, early stopping, 10% internal validation fraction
- random seed 42
- `RandomizedSearchCV` (default 4 candidates, 3-fold CV; increase with `--search-iterations` if desired)
- Search choices include hidden layers `(64, 32)`, `(100, 50)`, or `(128, 64)`, learning rate, L2 regularization `alpha`, and batch size.
- The inverse-transformed MLP output is clipped to the minimum/maximum price observed in the training split. This prevents extreme MLP extrapolation for atypical inputs; it also means the model should not be used outside the dataset's price domain.

All transformations and the search-selected model are saved together with `joblib` in `models/trained_model.joblib`.

## 7. Evaluation and visual outputs

`outputs/metrics.json` reports validation and test **MAE**, **MSE**, **RMSE**, and **R²** on the original USD scale, plus split sizes, selected features, price skewness, and chosen hyperparameters. The image-ablation JSON records the same-split comparison. Training creates:

- `actual_vs_predicted.png`
- `residual_plot.png`
- `error_distribution.png`
- `loss_curve.png` (MLP training loss and internal validation R² when exposed by sklearn)
- `price_distribution.png` (raw price and `log1p(price)`)

`--evaluate` re-creates the metrics and plots on the same deterministic holdout split using the saved model. Since metrics depend on the fitted estimator and dependency versions, use the generated `metrics.json` as the authoritative run result.

## 8. Install dependencies

Python 3.11 or newer is recommended (the bundled model was trained with Python 3.12). From this project folder:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The included `data/socal2.csv` is sufficient for standard tabular training; no Kaggle account or authentication is required. If the CSV is removed, the loader attempts to download the public Kaggle archive automatically. The optional image path requires the full archive (approximately 390 MB). If automatic Kaggle download is unavailable, download the dataset from Kaggle, put `socal2.csv` directly in `data/`, and for image use put the JPEGs named `<image_id>.jpg` in `data/socal_pics/`.

## 9. Run the complete application

After installing dependencies and extracting the project, the complete application starts with **one command**:

```bash
python main.py
```

The application loads `data/socal2.csv` (and downloads it from Kaggle if missing), then:

1. Loads the cached `models/trained_model.joblib` when it exists and prints **“Loading existing trained model...”**. It asks `Retrain model? (y/n):`; press `n` (or Enter) to reuse the included model, or `y` to train again.
2. If no compatible saved model exists, it automatically preprocesses, splits, tunes, and trains `MLPRegressor` once, then saves the full preprocessing/model pipeline.
3. Evaluates the model on the fixed validation/test split and displays test **MAE**, **RMSE**, and **R²**.
4. Enters interactive prediction mode. Prompts are generated from the trained model's actual feature list and dataset types. Numeric fields are validated against observed ranges. For categories such as city, the program shows common choices; enter a choice number or type a city name.
5. Displays the price in USD with cents and asks `Predict another house? (y/n):`. Repeated predictions reuse the same loaded/trained model; there is no retraining between houses. Enter `n` to exit.

For this inspected dataset/model the interactive features are **Bedrooms**, **Bathrooms**, **Square Feet**, and **City**. The exact fitted preprocessing pipeline is reused for every prediction. Unknown city names are handled safely by the one-hot encoder.

Optional developer commands remain available:

```bash
python main.py --train                 # train and save, then exit
python main.py --evaluate              # evaluate saved model, then exit
python main.py --predict               # prediction-only interactive mode
python main.py --train --use-images    # experimental handcrafted image feature path
python main.py --train --search-iterations 6 --cv 3
```

## 10. Project structure

```text
house-price-mlp/
├── data/
│   ├── README.md
│   └── socal2.csv
├── models/
│   └── trained_model.joblib       # included; automatically created if missing
├── outputs/
│   ├── metrics.json               # created by --train / --evaluate
│   ├── image_ablation_metrics.json # measured tabular-vs-image comparison
│   └── *.png                      # evaluation plots
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── image_features.py
│   ├── prediction_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── main.py
```

## 11. Limitations and future work

- The source dataset's Kaggle page reports an unknown license; check its current terms before redistribution or commercial use. The JPEG collection is not included in this project.
- Prices are historical/listed dataset values and are not a professional appraisal or current market quote.
- `street` is excluded, and only 415 city categories are represented; unseen cities remain processable but the model has no learned city-specific category effect for them.
- Outlying source values (for example, unusually large bath or square-foot counts) are retained rather than silently removed; inspect inputs before relying on estimates.
- Because the estimator is nonlinear and may extrapolate unusually for rare inputs, returned prices are constrained to the training split's observed target range; the model cannot predict below/above that range.
- Images are exterior-only. The tested HOG/color variant underperformed the tabular MLP on this held-out split; `--use-images` remains available to test different handcrafted features or larger tuning budgets without pretrained deep-learning models.
- Possible improvements include location coordinates, temporal sale information, grouped/geographic validation, richer structured property features, image-vs-tabular ablation experiments, and larger tuning budgets.

## 12. Viva summary

This project demonstrates **supervised learning**: labeled houses provide predictor variables and a known price target. It is a **regression** task because the target is continuous. A fixed train/validation/test split is used; a `ColumnTransformer` performs imputation, scaling, and city encoding; `RandomizedSearchCV` tunes `MLPRegressor`; `log1p` stabilizes the skewed target and `expm1` restores dollar predictions. MAE, MSE, RMSE, and R² measure held-out performance, and the saved pipeline powers the interactive prediction interface.
