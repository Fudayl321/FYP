from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset (make sure this path points to your CSV file)
data = pd.read_csv('US Stock Market Dataset.csv')

# Assuming you have already loaded your data into the DataFrame 'data'
data.ffill(inplace=True)  # Updated forward fill

# Convert the 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Check if the Date column exists before converting it
if 'Date' in data.columns:
    # Convert the 'Date_column' to Datetime
    data['Date'] = pd.to_datetime(data['Date'])
else:
    print("Warning: 'Date' not found in the dataset")

# Drop non-numeric columns (like Date columns) before standardizing
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Standardize numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Add back the non-numeric columns (e.g., Date_column) to the standardized data
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)

# Optionally, add back any non-numeric columns (such as Date columns) if needed
if 'Date' in data.columns:
    scaled_df['Date'] = data['Date']

# Output the first few rows to ensure correctness
print(scaled_df.head())

from sklearn.ensemble import IsolationForest

# Fit the model
model = IsolationForest(contamination=0.01)
model.fit(scaled_data)

# Predict anomalies (-1 for anomalies, 1 for normal)
predictions = model.predict(scaled_data)

# Filter anomalies
anomalies = data[predictions == -1]
