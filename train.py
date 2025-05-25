import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Excel file instead of CSV
data = pd.read_excel('big_five_dataset.xlsx', engine='openpyxl')

# Print column names
print("Loaded columns:")
print(data.columns)

feature_cols = [f'Q{i}' for i in range(1, 51)]
target_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

X = data[feature_cols]
y = data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("R² score:", model.score(X_test, y_test))

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as rf_model.pkl.")
