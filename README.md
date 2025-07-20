```markdown
# 💳 Transaction Risk Analyzer

**Transaction Risk Analyzer** is a fully interactive Streamlit dashboard that explains how an XGBoost classification model detects potentially fraudulent financial transactions. By leveraging SHAP (SHapley Additive exPlanations), the app translates complex model behavior into intuitive visualizations, making it easier to understand AI-driven fraud detection systems.

---

## 🎯 Purpose

This tool bridges technical transparency with user accessibility. Whether you're an ML practitioner, analyst, or someone curious about responsible AI, this project helps you:

- Understand the reasoning behind fraud predictions at both local (per-transaction) and global (model-wide) levels
- Interpret how specific features (amounts, balances, transaction types) influence model output
- Visualize SHAP explanations to demystify the "black box" nature of tree-based models

---

## 🧠 Technical Features

- ⚙️ **XGBoost Binary Classifier** trained on structured transactional data
- 💡 **SHAP Value Generation** using `shap.Explainer` and `shap.Explanation` objects
- 📈 **Model-Wide Explanation** with SHAP summary bar plots
- 🧠 **Per-Transaction Explanations** with ranked SHAP values and natural language insights
- 🧪 **Streamlit Frontend** for real-time user interaction
- 🧵 Modular architecture supporting precomputed SHAP values and stored model objects (`.pkl`)

---
<pre> ``` ├── app.py # Streamlit app for SHAP-based fraud explanation ├── style.css # Optional custom styling for Streamlit ├── sample_transactions.csv # Subset of labeled sample transactions ├── transactions.csv # Full dataset of financial transactions ├── shap_input.csv # Precomputed SHAP values ├── shap_explanations.csv # Additional SHAP breakdowns (optional) ├── xgb_model.json # Saved XGBoost model structure ├── label_encoder.pkl # Encoder for categorical transaction types ├── README.md # Project overview and instructions ├── LICENSE # MIT License ├── .gitignore # Git ignored files │ ├── explore_data.ipynb # Notebook for data exploration ├── model_training.ipynb # Notebook for model development ├── model_explainer.ipynb # Notebook for SHAP explanation generation ├── selecting_transactions.ipynb # Logic for selecting and saving demo samples │ ├── archive/ # Folder for archived files ├── eda/ # Folder for EDA-related scripts/notebooks └── venv/ # Python virtual environment ``` </pre>

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/transaction-risk-analyzer.git
cd transaction-risk-analyzer
````

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

---

## 📌 SHAP Setup (Optional)

If you're not using the precomputed `shap_input.csv`, you can dynamically generate SHAP values with:

```python
from shap import Explainer
import joblib

model = joblib.load("xgb_fraud_model.pkl")
explainer = Explainer(model)
shap_values = explainer(X_all)
```

---

## ⚖️ Model Info

* **Model**: XGBoost Classifier (`xgb.XGBClassifier`)
* **Metrics**: Tuned for precision-recall tradeoffs on imbalanced datasets
* **Features Used**:

  * `amount`, `step`, `oldbalanceOrg`, `newbalanceOrig`
  * `oldbalanceDest`, `newbalanceDest`, `encoded_type`

---

## 🧠 Why SHAP?

SHAP provides model-agnostic explanations grounded in cooperative game theory. It breaks down the contribution of each feature to the prediction, making the decision process interpretable and auditable.

---

## 👤 Author

Made with ❤️ by [Tanzil Hussain](https://www.linkedin.com/in/tanzilhussain)
Connect with me to chat about AI, responsible ML, and data-driven UX.

---

## 📄 License

MIT License — use with purpose and responsibility.

```
