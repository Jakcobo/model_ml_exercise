# E-Commerce Product Condition Classification

## Repository Structure

```
├── data_clean/
│   └── mla_eda.csv           # Cleaned dataset after EDA and transformations
├── best_model/
│   └── randomforest.pkl      # Serialized Random Forest model for inference
├── notebooks/
│   ├── 01_eda.ipynb  # Exploratory Data Analysis and feature engineering
│   └── 02_model.ipynb          # Model training, evaluation, and comparison
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 1. Project Overview

This project builds a classification pipeline to predict whether a product listing on MercadoLibre is **new** (`condition=0`) or **used** (`condition=1`). It covers:

* Data ingestion from JSONLines
* Comprehensive EDA and feature engineering
* Model training, tuning, and evaluation
* Model persistence for deployment

## 2. Dataset

* **Raw source:** `data/raw/MLA_100k.jsonlines` (100,000 records)
* **Cleaned file:** `data_clean/mla_eda.csv` after filtering, imputing, and transformations

### Key Columns

| Column               | Type     | Description                                                           |
| -------------------- | -------- | --------------------------------------------------------------------- |
| `condition`          | int      | Target: 0=new, 1=used                                                 |
| `listing_type_id_*`  | bool/int | One-hot flags for listing package levels (bronze, silver, free, etc.) |
| `price`              | float    | Listing price                                                         |
| `buying_mode`        | int      | Coded buying mode (e.g., purchase, auction)                           |
| `status`             | int      | Coded status of listing                                               |
| `initial_quantity`   | int      | Units initially available                                             |
| `available_quantity` | int      | Units currently available                                             |
| `free_shipping`      | bool     | Whether shipping is free                                              |
| `shipping_mode`      | int      | Coded shipping mode                                                   |
| `non_mp_pay_count`   | int      | Count of non-MercadoPago payment methods                              |

## 3. EDA & Feature Engineering

Executed in `01_EDA_and_Transformations.ipynb`:

1. **Schema inspection**: Ensured no missing values in key features.
2. **Column filtering**: Dropped uninformative fields (e.g., `subtitle`, `original_price`, `international_delivery_mode`).
3. **Type conversions**:

   * JSON→ numeric: extracted `latitude`/`longitude` from `geolocation` and imputed missing (\~5% records). But these columns was eliminated, beacause aren't important for model.

4. **Derived features**:

   * `non_mp_pay_count`: number of payment methods beyond MercadoPago. -> was eliminated
   * `pct_sold`: ratio sold/initial (later dropped due to redundancy). -> was eliminated
   * One-hot encoding for `listing_type_id`.

5. **Visual analysis**: Histograms, boxplots, ECDF for price; bar charts for categorical distributions; correlation heatmap.

## 4. Model Training & Evaluation

Detailed in `02_Model_Training.ipynb`:

1. **Train/Test Split**: 80/20 stratified on `condition`.
2. **Models**:

   * **Logistic Regression** with `StandardScaler`.
   * **Random Forest** without scaling.
3. **Evaluation metrics**:

   * **Logistic Regression**: ROC AUC = 0.756, Accuracy = 0.68, F1(new)=0.70, F1(used)=0.65
   * **Random Forest**: ROC AUC = 0.895, Accuracy = 0.81, F1(new)=0.82, F1(used)=0.81
4. **Selection**: Random Forest outperforms across all metrics and exhibits balanced precision/recall.

## 5. Model Persistence

* The final `RandomForestClassifier` is serialized with:

  ```python
  import joblib
  joblib.dump(model, 'best_model/randomforest.pkl')
  ```
* Ready for loading in production:

  ```python
  loaded_rf = joblib.load('best_model/randomforest.pkl')
  preds = loaded_rf.predict(new_X)
  ```


## 6. Dependencies

Key libraries:

* pandas, numpy
* scikit-learn
* seaborn, matplotlib
* joblib