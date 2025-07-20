# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

def load_data():
  # Set the path to the file you'd like to load
  file_path = "data/transactions.csv"

  # Load the latest version
  df = pd.read_csv(file_path)
  df['log_amount'] = np.log1p(df['amount'])
  return df

def eda(df):
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


  # numeric correlation to fraud
  correlation_with_fraud = df.corr(numeric_only=True)['isFraud'].drop('isFraud').sort_values(ascending=False).round(3)
  z = [correlation_with_fraud.values.tolist()]
  x = correlation_with_fraud.index.tolist()
  y = ['isFraud']

  fig = ff.create_annotated_heatmap(
    z=z,
    x=x,
    y=y,
    colorscale='Viridis',
    zmin=-1, zmax=1,
    showscale=True,
    annotation_text=[[f"{val:.2f}" for val in z[0]]]
  )
  fig.update_layout(title='Correlation Heatmap of Numeric Features')
  fig.write_html("correlation_heatmap.html")


if __name__ == '__main__':
    df = load_data()
    eda(df)