from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
data = pd.read_csv('US Stock Market Dataset.csv')

# Fill forward missing values in the entire dataset
data.ffill(inplace=True)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Impute missing numeric values with the mean
numeric_data = data.select_dtypes(include=['float64', 'int64'])
imputer = SimpleImputer(strategy='mean')
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data_imputed)

# Create a DataFrame for scaled data
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

# Fit the IsolationForest model
model = IsolationForest(contamination=0.01)
predictions = model.fit_predict(scaled_data)

# Add predictions to the DataFrame for easy plotting
scaled_df['anomaly'] = predictions

# Separate out the normal points (label == 1) and anomalies (label == -1)
normal = scaled_df[scaled_df['anomaly'] == 1]
anomalies = scaled_df[scaled_df['anomaly'] == -1]

# Visualize
plt.figure(figsize=(10, 6))

# Plot normal points
sns.scatterplot(data=normal, x=normal.iloc[:, 0], y=normal.iloc[:, 1], 
                label='Normal', marker='o', color='blue')

# Plot anomalies
sns.scatterplot(data=anomalies, x=anomalies.iloc[:, 0], y=anomalies.iloc[:, 1], 
                label='Anomalies', marker='x', color='red')

# Customize plot
plt.title('IsolationForest Anomalies Detection')
plt.xlabel('Feature 1')  # Replace with actual feature names if available
plt.ylabel('Feature 2')  # Replace with actual feature names if available
plt.legend()
plt.show()
