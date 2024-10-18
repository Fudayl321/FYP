from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
data = pd.read_csv('stock_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
