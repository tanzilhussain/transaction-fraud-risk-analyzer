import pandas as pd
from archive.explore_data import load_data
import shap
import joblib
df = pd.read_csv("data/transactions.csv")

# filter csv 
conditions = (
    (df['step'] == 58) & (df['amount'] == 561948.38) |
    (df['step'] == 125) & (df['amount'] == 222097.11) |
    (df['step'] == 137) & (df['amount'] == 5401.53) |
    (df['step'] == 614) & (df['amount'] == 33354.78) |
    (df['step'] == 283) & (df['amount'] == 2210702.71)
)

df_samples = df[conditions]

print(df_samples)

# reload model
model = joblib.load("xgb_fraud_model.pkl")
le = joblib.load("label_encoder.pkl")
df_samples['encoded_type'] = le.fit_transform(df_samples['type'])

df_samples.to_csv("data/sample_transactions.csv", index=False)
# features included in model
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest', 'encoded_type']
X_samples = df_samples[features]

# shap for sample
explainer = shap.Explainer(model)
shap_values = explainer(X_samples)
shap.plots.bar(shap_values)
# shap values array (rows, features)
print(type(shap_values))
shap_values_array = shap_values.values
shap_df = pd.DataFrame(shap_values_array, columns=features)
shap_df['prediction'] = model.predict(X_samples)
shap_df['actual'] = df_samples['isFraud'].values
shap_df['transaction_type'] = df_samples['type'].values
shap_df['amount'] = df_samples['amount'].values

# shap_df.to_csv("shap_explanations.csv", index=False)
# print(shap_df.head())
