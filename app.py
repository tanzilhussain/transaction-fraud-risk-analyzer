import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# load data
df_samples = pd.read_csv("sample_transactions.csv")
shap_df = pd.read_csv("shap_explanations.csv")

# features used in model
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'encoded_type']

# rename SHAP columns
shap_df.columns = [f"{col}_shap" for col in features] + ['prediction', 'actual', 'transaction_type']
df = pd.concat([df_samples.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

# custom CSS
try:
    with open("style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except:
    pass

# TITLE
st.title("Transaction Risk Analyzer")

st.markdown("""
Welcome to the Transaction Risk Analyzer, a tool that helps you uncover how AI models detect suspicious financial activity.

Scroll through examples of transactions the model flagged as potentially fraudulent, and learn which key factors influence the model's decisions to help you better understand risky patterns.

Whether you're a developer, analyst, or just curious about AI, this tool gives you a behind-the-scenes look at how machine learning models think about fraud.
""")
st.markdown("<br>", unsafe_allow_html=True)
# basic example
st.header("Suspicious Transaction Example")
example = df[df['prediction'] == 1].iloc[0]

st.markdown(f"""
- **Type**: `{example['type']}`
- **Amount**: `${example['amount_shap']:,.2f}`
- **Sender Balance**: `${example['oldbalanceOrg']:,.2f}` → `${example['newbalanceOrig']:,.2f}`
- **Receiver Balance**: `${example['oldbalanceDest']:,.2f}` → `${example['newbalanceDest']:,.2f}`
- **Model Prediction**: 🚨 Fraud
""")

# explanation toggle
if st.toggle("🔎 Explain this prediction"):
    # SHAP Explanation object for Step 2
    shap_vals = example[[f"{f}_shap" for f in features]].values
    input_features = example[features].values
    expl = shap.Explanation(values=shap_vals,
                            base_values=0,
                            data=input_features,
                            feature_names=features)

    st.write("### SHAP Explanation")
    fig, ax = plt.subplots()
    shap.plots.bar(expl, show=False)
    st.pyplot(fig)

    # Auto-generated explanation
    top_features = expl.abs.values.argsort()[-3:][::-1]
    top_feature_names = [features[i] + "_shap" for i in top_features]
    reason_texts = {
        'amount_shap': "an unusually large transaction amount",
        'oldbalanceOrg_shap': "a low sender balance",
        'newbalanceOrig_shap': "a suspicious change in sender balance",
        'oldbalanceDest_shap': "a suspicious change in receiver balance",
        'newbalanceDest_shap': "a sharp increase in receiver balance",
        'step_shap': "an unusual transaction time",
        'encoded_type_shap': f"a high-risk transaction type: {example['transaction_type']}"
    }

    st.write("💡 The model flagged this transaction likely due to:")
    for feat in top_feature_names:
        st.markdown(f"- {reason_texts.get(feat, feat)}")

st.markdown("<br>", unsafe_allow_html=True)

# looking at other transactions
st.header("📚 Learn from Other Examples")

# create options with placeholder
transaction_options = ["-- Select a Transaction --"] + [f"Transaction {i + 1}" for i in df.index]
selected_label = st.selectbox("Pick a Sample Transaction", transaction_options)

if selected_label != "-- Select a Transaction --":
    txn_index = int(selected_label.split(" ")[-1]) - 1
    txn = df.loc[txn_index]

    st.markdown(f"### Prediction: {'🚨 Fraud' if txn['prediction'] == 1 else '✅ Not Fraud'}")

    with st.expander("📋 Transaction Details"):
        st.dataframe(txn.to_frame().T)

    # SHAP explanation for selected transaction
    shap_vals_txn = txn[[f"{f}_shap" for f in features]].values
    input_txn_features = txn[features].values
    expl_txn = shap.Explanation(values=shap_vals_txn,
                                base_values=0,
                                data=input_txn_features,
                                feature_names=features)

    st.write("### SHAP Explanation")
    fig2, ax2 = plt.subplots()
    shap.plots.bar(expl_txn, show=False)
    st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)
st.header("🌐 How the Model Thinks (Overall)")
st.markdown("This plot shows which features most influence the model **on average** across all transactions.")

# recreate SHAP values
X_all = df_samples[features]
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")
explainer = shap.Explainer(model)
shap_values_all = explainer(X_all)
pd.DataFrame(shap_values_all.values, columns=features).to_csv("shap_input.csv")

# generate the SHAP summary plot
fig_summary, ax = plt.subplots()
shap.summary_plot(shap_values_all, X_all, plot_type="bar", show=False)
st.pyplot(fig_summary)

