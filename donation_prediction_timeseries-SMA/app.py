import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('./donations_data.csv')

# Set the index to 'Members Name' for easier access
df.set_index('Members Name', inplace=True)

# Transpose the DataFrame to have months as rows and members as columns
df = df.transpose()

# Convert all values to numeric, coercing errors and dropping non-numeric columns
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with 0 or use another strategy if preferred
df.fillna(0, inplace=True)

# Set frequency for time series data (monthly)
df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='M')

# Function to calculate Simple Moving Average and forecast next month's value for each member
def forecast_next_month(member_data, window=3):
    # Calculate the moving average
    moving_avg = member_data.rolling(window=window).mean()
    
    # Forecast next month as the last moving average value
    return moving_avg.iloc[-1]

# Streamlit application layout
st.title("Donation Prediction using Simple Moving Average")

# Input for member name
member_name = st.selectbox("Select Member Name", df.columns)

if st.button("Predict"):
    # Get prediction for selected member
    prediction = forecast_next_month(df[member_name])
    
    # Display result
    st.write(f"The predicted donation for {member_name} for next month is: **{prediction:.2f}**")
    
    # Plotting the actual values and predicted value
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[member_name], marker='o', label='Actual Donations')
    plt.axhline(y=prediction, color='r', linestyle='--', label='Predicted Donation for Next Month')
    
    plt.title(f'Donation Trend for {member_name}')
    plt.xlabel('Month')
    plt.ylabel('Donation Amount')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Show plot in Streamlit
    st.pyplot(plt)

