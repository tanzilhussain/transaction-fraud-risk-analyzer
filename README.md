# 💳 Transaction Fraud Risk Analyzer

**Transaction Fraud Risk Analyzer** is an interactive Streamlit dashboard built to **explain how an XGBoost model detects potentially fraudulent transactions**. By combining **SHAP (SHapley Additive exPlanations)** with an intuitive UI, this project transforms a “black box” fraud detection model into a transparent, educational tool.

---
## 🔍 Live Demo

Explore the interactive dashboard here:  
[🚨 Transaction Fraud Risk Analyzer](https://transaction-fraud-risk-analyzer.streamlit.app/)

This app uses explainable AI (SHAP) to detect potential fraud in financial transactions and educate users about risky patterns.

---

## 📘 What I Learned

This project taught me far more than just model training.

🔎 **Interpretability vs. Accuracy**  
While tuning my XGBoost classifier, I realized that making predictions is only half the battle, as helping users **understand** those predictions is what builds trust. That’s where SHAP came in. Learning how to visualize feature contributions and explain model behavior pushed me to think like both a data scientist and a UX designer.

⚖️ **Balancing Risk in an Imbalanced Dataset**  
Fraud detection is inherently skewed. I learned to work with **imbalanced data**, carefully optimizing for **precision and recall** rather than accuracy. This meant adapting my threshold, using confusion matrices, and ensuring my classifier could detect rare but critical fraud cases.

📊 **From Code to Communication**  
SHAP’s math is beautiful, but for everyday users, it can be overwhelming. Building this as a **dashboard instead of a notebook** forced me to think through *how to translate model logic into plain language*. I added dynamic reasons, simple toggles, and friendly charts that expose the "why" without exposing the model to manipulation.

---

## 🎯 Purpose & Reflection

Fraud detection models are often treated as black boxes. While building this, I wanted to answer:
- **What makes a transaction look “suspicious” to a model?**
- **How can we demystify AI predictions without revealing exploitable details?**

This project started as a technical experiment with XGBoost and imbalanced datasets. But as I built SHAP explanations, I realized how important *interpretability* is, not just for developers, but for anyone impacted by AI-driven financial decisions.

---

## 🧠 Technical Features

- ⚙️ **XGBoost Binary Classifier** trained on real-world-like transactional data  
- 💡 **SHAP Value Generation** (`shap.Explainer` & `shap.Explanation`) to quantify feature contributions  
- 📈 **Model-Wide Explanations** via SHAP summary plots (global feature importance)  
- 🔍 **Per-Transaction Explanations** with ranked SHAP values and human-readable reasoning  
- 🎛 **Streamlit Dashboard** for interactive exploration of flagged vs. safe transactions  
- 🧵 **Modular Backend** with precomputed SHAP values and serialized model objects (`.pkl`)  

---

## 🌍 Broader Impact

Fraud detection isn't just a technical challenge — it's about **trust**. With tools like this, I hope to:
- Show users **why** a transaction might be flagged, encouraging **awareness of risky behaviors**  
- Promote **responsible AI**, where explainability is prioritized alongside accuracy  
- Provide a template for **educational dashboards** that teach, rather than just predict  

---

## ⚖️ Model Info

* **Model**: XGBoost Classifier (`xgb.XGBClassifier`)
* **Focus**: Precision-recall optimization on an imbalanced dataset
* **Features Used**:

  * `amount`, `step`, `oldbalanceOrg`, `newbalanceOrig`
  * `oldbalanceDest`, `newbalanceDest`, `encoded_type`

---

## 🧠 Why SHAP?

SHAP applies **game theory** to explain predictions, showing the **marginal contribution** of each feature. This makes the model’s reasoning **transparent and auditable**, even for tree-based ensembles like XGBoost.

---
## 📊 Dataset

This project uses the [**Credit Card Transactions Dataset**](https://www.kaggle.com/datasets/kelvinobiri/credit-card-transactions) by Kelvin Obiri, available on Kaggle under the [MIT License](https://opensource.org/licenses/MIT).

The dataset contains anonymized financial transaction records, including:

- `type`: Transaction type (e.g., CASH_OUT, TRANSFER)
- `amount`: Amount involved in the transaction
- `oldbalanceOrg` / `newbalanceOrig`: Sender’s balance before and after
- `oldbalanceDest` / `newbalanceDest`: Receiver’s balance before and after
- `isFraud`: Target variable (1 if fraudulent, 0 otherwise)

> 📌 **Disclaimer**: The dataset is anonymized and intended for educational and research purposes only. It does not contain any real personal financial information.

---

## 👤 Author

Created with ❤️ by [Tanzil Hussain](https://www.linkedin.com/in/tanzilhussain)
Passionate about **responsible AI**, **explainability**, and **data-driven UX**.

---
