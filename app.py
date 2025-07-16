import streamlit as st

st.title("Financial Transaction Fraud Detection Dashboard")
with st.form("simulation"):
    # transaction type
    st.selectbox('Transaction Type: ', ('Transfer', 'Cashing Out', 'Cashing In', 'Debit', 'Payment'))
    st.number_input('Enter Transaction Amount: ')
    sender_curr_balance = st.number_input('Sender Account Balance (before transaction): ')
    receiver_curr_balance = st.number_input('Receiver Account Balance (before transaction): ')
    submitted = st.form_submit_button("Submit")
    # if submitted:
    # ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT']