import streamlit as st
from sklearn.preprocessing import LabelEncoder

with open("style.css") as css:
     st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='dashboard_title'>Financial Transaction Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

with st.form("simulation"):
    # transaction type
    type = st.selectbox('Transaction Type: ', ('Transfer', 'Cashing Out', 'Cashing In', 'Debit', 'Payment'))
    st.number_input('Enter Transaction Amount: ')
    sender_curr_balance = st.number_input('Sender Account Balance (before transaction): ')
    receiver_curr_balance = st.number_input('Receiver Account Balance (before transaction): ')
    submitted = st.form_submit_button("Submit")
    if submitted:
        # handling categorical data 
        le = LabelEncoder()
        encoded_type = le.transform(type)
    # ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT']