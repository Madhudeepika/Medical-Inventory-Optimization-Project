import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load your cleaned dataset
# Replace 'your_dataset.csv' with your actual file name
df = pd.read_csv(r"A:\Downloads\preprocessedData (1).csv")

# Assuming your dataset has columns like 'Dateofbill', 'Quantity', and other relevant features

# Convert 'Dateofbill' column to datetime format
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])

# Set 'Dateofbill' as the index
df.set_index('Dateofbill', inplace=True)

# Additional preprocessing if needed
# ...

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Define the SARIMAX model
order = (1, 1, 1)  # Replace with your chosen order
seasonal_order = (1, 1, 1, 12)  # Replace with your chosen seasonal order

# Train the SARIMAX model
sarimax_model = SARIMAX(train['Quantity'], order=order, seasonal_order=seasonal_order)
sarimax_fit = sarimax_model.fit()

# Save the trained model using joblib
joblib.dump(sarimax_fit, 'sarimax_model.pkl')

# Make predictions on the test set
predictions = sarimax_fit.get_forecast(steps=len(test))
forecasted_values = predictions.predicted_mean

# Evaluate the model
mae = mean_absolute_error(test['Quantity'], forecasted_values)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Save the preprocessor (if applicable)
#preprocess_fit = YourPreprocessor.fit(your_data)
#joblib.dump(preprocess_fit, 'preprocess_fit.pkl')

# Save the SARIMAX model
joblib.dump(sarimax_fit, 'sarimax_model.pkl')



