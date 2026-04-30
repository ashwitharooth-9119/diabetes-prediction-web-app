import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DPF', 'Age', 'Outcome'
]

data = pd.read_csv(url, names=columns)

# Replace 0 with median
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    data[col] = data[col].replace(0, data[col].median())

# Split
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = LogisticRegression()
model.fit(X_scaled, y)

# Save
pickle.dump(model, open('diabetes_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model saved successfully")