# 🛡️ Credit Card Fraud Detection — Databricks ML Pipeline

A full end-to-end machine learning pipeline built on Databricks for detecting fraudulent credit card transactions. The project follows the **Medallion Architecture** (Bronze → Silver → Gold) for data engineering, and uses a **Gradient Boosted Trees (GBT)** classifier tracked via **MLflow** for model training and evaluation.

---

## 📁 Project Structure

```
├── 01_Bronze_Ingestion.py          # Raw data ingestion from CSV to Delta table
├── 02_Silver_Transformation.py     # Data cleaning, EDA, and outlier handling
├── 03_Gold_Feature_Engineering.py  # Class weighting and feature assembly
├── 04_Model_Training.py            # GBT model training with MLflow tracking
└── report.py                       # Summary report and business impact notes
```

---

## 🏗️ Architecture Overview

This project uses the **Lakehouse Medallion Architecture**, where data flows through three progressively refined layers before reaching model training.

```
Raw CSV  →  Bronze (Delta)  →  Silver (Delta)  →  Gold (Delta)  →  ML Model
```

---

## 📓 Notebook Breakdown

### 1️⃣ Bronze — Data Ingestion (`01_Bronze_Ingestion.py`)

This is the entry point of the pipeline. It reads the raw credit card transaction data from a CSV file stored in a Databricks Volume and saves it as a Delta table without any modifications.

- **Source:** `/Volumes/workspace/default/fraud_volume/creditcard.csv`
- **Output Table:** `fraud_bronze`
- Schema is inferred automatically from the CSV headers.

---

### 2️⃣ Silver — Transformation & EDA (`02_Silver_Transformation.py`)

This notebook is where the heavy lifting of data cleaning and exploratory analysis happens. Key steps include:

- **Null check** across all columns to ensure data integrity
- **Type casting** — the `Class` column is explicitly cast to integer
- **Outlier detection** using the IQR method on the `Amount` column:
  - Lower and upper bounds are calculated using the 25th and 75th percentiles
  - Records are split into `normal_df` and `outlier_df`
- **Exploratory Visualizations:**
  - Class distribution (Fraud vs. Legitimate) — confirms heavy class imbalance
  - Amount distribution before and after outlier removal
  - Box plot of `Amount` grouped by `Class`
  - Log-transformed amount distribution — reduces skew and improves model stability
  - Overlapping histogram of Fraud vs. Legitimate log-amounts
- **Output Table:** `fraud_silver`

> 💡 Key insight from EDA: The dataset is highly imbalanced, most transactions are low-value, and log-transformation of `Amount` significantly reduces skew — making it more suitable for linear models.

---

### 3️⃣ Gold — Feature Engineering (`03_Gold_Feature_Engineering.py`)

This layer prepares the data in a format ready for ML training.

- **Class imbalance handling:** A `weight` column is added to each record. Fraud transactions (`Class = 1`) receive a weight equal to the legitimate-to-fraud ratio, while legitimate transactions get a weight of 1.
- **Feature assembly:** All columns except `Class` and `weight` are combined into a single `features` vector using Spark MLlib's `VectorAssembler`.
- **Output Table:** `fraud_gold`

---

### 4️⃣ Model Training (`04_Model_Training.py`)

The final modeling notebook trains a **Gradient Boosted Trees (GBT)** classifier and logs everything to MLflow.

- **Train/Test split:** 80/20 with seed 42 for reproducibility
- **MLflow experiment:** `/Fraud_Detection_Project`
- **Model:** `GBTClassifier` with 50 iterations
- **Evaluation metrics computed:**
  - ROC-AUC (`BinaryClassificationEvaluator`)
  - F1 Score
  - Weighted Precision
  - Weighted Recall
- The trained model is logged to MLflow under the artifact name `gbt`.

---

### 📊 Report (`report.py`)

A summary notebook documenting the final model selection rationale and estimated business impact.

- **Best Model:** Gradient Boosted Trees
- **Why GBT?** Highest ROC-AUC, best fraud recall, and balanced precision
- **Business Impact:** Detecting ~92% of fraud cases at an average fraud value of $1,000 per case translates to potentially millions of dollars saved annually.

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Platform | Databricks |
| Data Format | Delta Lake |
| Processing | Apache Spark (PySpark) |
| ML Framework | Spark MLlib |
| Experiment Tracking | MLflow |
| Visualizations | Matplotlib |
| Language | Python |

---

## 🚀 How to Run

1. Upload `creditcard.csv` to the Databricks Volume at:
   `/Volumes/workspace/default/fraud_volume/creditcard.csv`

2. Run the notebooks **in order**:
   ```
   01_Bronze_Ingestion.py
   02_Silver_Transformation.py
   03_Gold_Feature_Engineering.py
   04_Model_Training.py
   ```

3. View model metrics and artifacts in the **MLflow UI** under the experiment `/Fraud_Detection_Project`.

---

## 📈 Model Performance

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary metric — measures ability to distinguish fraud from legitimate transactions |
| F1 Score | Balances precision and recall — critical for imbalanced datasets |
| Weighted Precision | How many flagged transactions are actually fraudulent |
| Weighted Recall | How many actual fraud cases were successfully caught |

> The GBT model was selected as the best performer across all four metrics.

---

## 📝 Notes

- The dataset is **highly imbalanced** — class weights are used during training to prevent the model from simply predicting "legitimate" for every transaction.
- Log-transformation of the `Amount` feature is applied during EDA to understand its distribution, though the actual transformation is handled at the feature engineering stage.
- MLflow artifacts are stored in `/Volumes/workspace/default/fraud_volume/mlflow_tmp`.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change or improve.

---

## 📄 License

This project is intended for educational and research purposes.
