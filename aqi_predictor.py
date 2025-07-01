import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/air_quality.csv')

# Drop rows with missing AQI values
df = df.dropna(subset=['AQI'])

# Drop irrelevant columns
df = df.drop(['City', 'Date'], axis=1)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and Target
X = df.drop('AQI', axis=1)
y = df['AQI']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Test prediction on new data
sample = X_test.iloc[0:1]
predicted_aqi = model.predict(sample)
print(f"Predicted AQI for sample input: {predicted_aqi[0]:.2f}")
