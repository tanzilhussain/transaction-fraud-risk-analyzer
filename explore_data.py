# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import plotly.express as px
import numpy as np
import pandas as pd
# Set the path to the file you'd like to load
file_path = "transactions.csv"

# Load the latest version
df = pd.read_csv(file_path)

# exploration
print("First 5 records:", df.sample(n=100))
print(df['isFraud'].value_counts())
print(f"\nFraud percentage: {df['isFraud'].mean() * 100:.4f}%")
print(df.info())
print(df.describe())

## making graphs and charts

# transaction types
type_counts = df['type'].value_counts().reset_index()
print(type_counts)
type_counts.columns = ['Transaction Type', 'Count']
fig = px.bar(type_counts, x = 'Transaction Type', y = 'Count', title="Transaction Type Distribution", text='Count')
fig.write_html("transaction_types.html")

# fraud rate by transaction type
fraud_rate = df.groupby('type')['isFraud'].mean().reset_index()
fraud_rate.columns = ['Transaction Type', 'Fraud Rate']
fig = px.bar(fraud_rate, x = 'Transaction Type', y = 'Fraud Rate', title="Fraud Rate by Transaction Type", text='Fraud Rate')
fig.write_html("fraud_rate.html")

# transaction amounts
df['log_amount'] = np.log1p(df['amount'])
fig = px.histogram(df, x='log_amount', nbins = 100, title="Log-Scaled Distribution of Transaction Amounts")
fig.update_layout(xaxis_title = 'Log(1 + Amount)', yaxis_title = 'Count')
fig.write_html("transaction_amount.html")